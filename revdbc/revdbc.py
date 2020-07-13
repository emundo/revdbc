from enum import Enum, auto
import logging
import os
import re
from typing import cast, Dict, List, NamedTuple, Optional, Set, Tuple, Union
import warnings

import cantools
import cantools.subparsers.dump
import numpy as np
from scipy.spatial.distance import minkowski
import sklearn
from sklearn import cluster
from sklearn import svm
from typing_extensions import Literal

from .maybe_open import Openable, maybe_open

warnings.filterwarnings(
    "ignore",
    message="Liblinear failed to converge, increase the number of iterations.",
    category=sklearn.exceptions.ConvergenceWarning,
    module=sklearn.__name__
)

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in true_divide",
    category=RuntimeWarning,
    module=np.__name__
)

LOGGER = logging.getLogger(__name__)

CANDUMP_ROW_REGEX = re.compile(r"^\((\d+\.\d+)\) (\w+) ([0-9A-F]{3,8})#([0-9A-F]*)\s*$")

SizedBitsNP = np.dtype([
    ("size", np.uint8),  # The number of valid bytes in `bits`.
    ("bits", np.uint64)  # Can hold up to 64 bits of data.
])

# Information about the source device/interface is not stored, all rows are assumed to belong to the
# same device/interface.
CandumpRowNP = np.dtype([
    ("timestamp", np.float64),  # The timestamp with 64 bit floating point precision
    ("identifier", np.uint32),  # The identifier (11/29 bits), packed into a 32 bit integer
    ("data", SizedBitsNP)       # The packet body
])


class FieldType(Enum):
    MULTIPLEXER = auto()
    MULTIPLEXED = auto()
    CONST = auto()
    VALUE = auto()
    UNKNOWN = auto()


class FieldEndianness(Enum):
    BIG = auto()
    LITTLE = auto()
    UNKNOWN = auto()

    def to_cantools(
        self,
        unknown: Union[Literal["little_endian"], Literal["big_endian"]] = "little_endian"
    ) -> Literal["little_endian", "big_endian"]:
        """
        Args:
            unknown: The value to return for the UNKNOWN enum value. Defaults to "little_endian".

        Returns:
            The string representation of this endianness as expected by cantools (i.e. "big_endian" or
            "little_endian").
        """

        if self is FieldEndianness.LITTLE:
            return "little_endian"

        if self is FieldEndianness.BIG:
            return "big_endian"

        return unknown


class Field(NamedTuple):
    lsb_anchor: int
    msb_anchor: int
    size: int
    endianness: FieldEndianness
    type: FieldType


class AnalysisResult(NamedTuple):
    restored_dbc_file: str
    restored_dbc: cantools.db.Database


class NotEnoughData(Exception):
    pass


class AnalysisFailed(Exception):
    pass


def load_candump(candump: Openable) -> np.ndarray:
    """
    Args:
        candump: Either a path-like object pointing to the candump file to load, or an opened readable
            file-like object representing the file.

    Returns:
        The loaded/parsed candump file.
        The `np.ndarray` contains elements of type `CandumpRowNP`.

    Raises:
        OSError: if any interaction with the file system failed.
    """
    entries = []
    with maybe_open(candump, "r") as f:
        for row in f:
            match = CANDUMP_ROW_REGEX.match(row)
            if match is None:
                LOGGER.warning("Skipping malformed row in candump file %s: %s", f.name, row)
                continue

            timestamp = np.array(match.group(1), dtype=np.float64)
            identifier = np.array(int(match.group(3), 16), dtype=np.uint32)

            # Swap the bytes after reading them to have bit indexing compatible with cantools.
            body = np.array(int(match.group(4).ljust(16, "0"), 16), dtype=np.uint64).byteswap()

            # Get the size of the body by counting hex digits in the candump text and rounding up to the next
            # full byte.
            body_size = np.array((len(match.group(4)) * 4 + 7) // 8, dtype=np.uint8)

            body_np = np.array((body_size, body), dtype=SizedBitsNP)

            entries.append(np.array((timestamp, identifier, body_np), dtype=CandumpRowNP))

    return np.array(entries, dtype=CandumpRowNP)


def analyze_identifier(
    identifier: int,
    bodies: np.ndarray,
    size: int,
    output_directory: str,
    suffix: Optional[str] = None
) -> AnalysisResult:
    """
    Args:
        identifier: The identifier that `bodies` belong to. Used only for logging/rendering purposes and to
            identify the analysis.
        bodies: A (non-empty) NumPy array containing CAN packet bodies belonging to the same identifier. The
            `np.ndarray` contains elements of type `np.uint64`.
        size: The number of bytes in each body. All bodies must have the same byte size.
        output_directory: The directory to save the results of the analysis to. Must exist.
        suffix: Suffix to append to all generated file names (but in front of the extensions).

    Returns:
        The result of the analysis.

    Raises:
        OSError: if any interaction with the file system failed.
        AnalysisFailed: if the analysis failed with an unrecoverable error.
    """

    # Find predictable fields in the CAN packet bodies. Predictable fields are possibly multiplexer
    # fields.
    muxes = _find_predictable_fields(bodies, size)

    # Constant fields are very much predictable too, but don't offer a lot of value for
    # demultiplexing.
    muxes = set(filter(lambda mux: mux.type is not FieldType.CONST, muxes))

    LOGGER.debug("Potential muxes:")
    for mux in muxes:
        LOGGER.debug("%s", mux)

    top_level_analyzed = set()
    mux_level_analyzed: Optional[Dict[int, Set[Field]]] = None
    mux_found = False

    # Demultiplex the bodies by grouping them by their multiplexer field values
    for mux in set(muxes):
        LOGGER.debug("-" * 100)
        LOGGER.debug("Demuxing and analyzing %s", mux)

        demuxed_analysis_results = {}
        demuxed = _demux_by(bodies, mux)
        for mux_value, demuxed_bodies in demuxed.items():
            LOGGER.debug("*" * 100)
            LOGGER.debug("Mux %s value %s", mux, mux_value)

            try:
                demuxed_analysis_results[mux_value] = _solve_overlaps(analyze_demultiplexed(
                    demuxed_bodies,
                    size * 8  # Pass the size in bits
                ), mux, size)
            except NotEnoughData as e:
                LOGGER.debug("Data too sparse: %s", e)

        if len(demuxed_analysis_results) < 2:
            LOGGER.info(
                "Data too sparse, couldn't demultiplex/analyze %s. Assuming it is not a multiplexer"
                " field.",
                mux
            )
            muxes.remove(mux)
            continue

        # The next step is to find out which fields seem to depend on the value of the multiplexer
        # and which don't. To do so, the code finds fields that are classified the same way for all
        # multiplexer values.
        detected_fields = [ x.fields for x in demuxed_analysis_results.values() ]
        detected_fields = list(filter(lambda x: len(x) > 0, detected_fields))

        if len(detected_fields) == 0:
            LOGGER.debug(
                "No fields detected when demultiplexing %s. Assuming it is not a multiplexer"
                " field.",
                mux
            )
            muxes.remove(mux)
            continue

        all_fields = set.union(*detected_fields)
        non_multiplexed_fields = set.intersection(*detected_fields)
        multiplexed_fields = all_fields - non_multiplexed_fields

        LOGGER.debug("~" * 100)
        # If all of the fields are independent of the multiplexer field value, it is probably not a
        # multiplexer
        if non_multiplexed_fields == all_fields:
            LOGGER.debug(
                "Mux %s does not influence the packet body. Assuming it is not a multiplexer"
                " field.",
                mux
            )
            muxes.remove(mux)
        else:
            if mux_found:
                raise AnalysisFailed("Found multiple multiplexer fields, analysis not possible.")
            mux_found = True

            def is_multiplexed_field(field: Field, muxed_fields: Set[Field] = multiplexed_fields) -> bool:
                return field in muxed_fields

            top_level_analyzed = non_multiplexed_fields
            mux_level_analyzed = {
                key: set(filter(is_multiplexed_field, value.fields))
                for key, value in demuxed_analysis_results.items()
            }

    LOGGER.debug("-" * 100)
    LOGGER.debug("Muxes:")
    for mux in muxes:
        LOGGER.debug("%s", mux)
    LOGGER.debug("-" * 100)

    if len(muxes) == 0:
        LOGGER.debug("No multiplexers, running global analysis:")
        # In case no multiplexers were detected, perform analysis on the untouched bodies
        analysis_result = analyze_demultiplexed(bodies, size * 8)  # Pass the size in bits

        top_level_analyzed = analysis_result.fields

    restored_dbc_info = _restore_dbc(
        int(identifier),
        size,
        top_level_analyzed,
        mux_level_analyzed,
        output_directory,
        suffix
    )

    return AnalysisResult(
        restored_dbc_file=restored_dbc_info[0],
        restored_dbc=restored_dbc_info[1]
    )


def _find_predictable_fields(bodies: np.ndarray, size: int) -> Set[Field]:
    """
    Args:
        bodies: A (non-empty) NumPy array containing CAN packet bodies belonging to the same identifier. The
            `np.ndarray` contains elements of type `np.uint64`.
        size: The number of bytes in each body. All bodies must have the same byte size.

    Returns:
        The set of predictable fields that were found.
    """

    # Find bit positions whose flips seem to be predictable
    predictable_fields = []
    for index in range(size * 8):
        # The endianness is unknown for single-bit fields
        field = Field(
            lsb_anchor=index,
            msb_anchor=index,
            size=1,
            endianness=FieldEndianness.UNKNOWN,
            type=FieldType.UNKNOWN
        )

        # Constant fields are predictable
        if _is_field_constant(bodies, field):
            predictable_fields.append(field._replace(type=FieldType.CONST))
            continue

        # If the field is not constant, perform the SVM prediction test
        if _is_field_predictable(bodies, field):
            predictable_fields.append(field._replace(type=FieldType.MULTIPLEXER))
            continue

    LOGGER.debug("-" * 100)
    LOGGER.debug("Predictable single-bit fields:")
    for predictable_field in predictable_fields:
        LOGGER.debug("%s", predictable_field)

    # Try to merge constant/predictable fields to build larger ones
    predictable_field_index = 0
    while predictable_field_index < len(predictable_fields) - 1:
        this_predictable_field = predictable_fields[predictable_field_index]

        LOGGER.debug("-" * 100)
        LOGGER.debug("Finding a merging buddy for %s", this_predictable_field)

        for other_predictable_field in predictable_fields[predictable_field_index + 1:]:
            merged_field = _merge_fields(this_predictable_field, other_predictable_field)
            if merged_field is None:
                LOGGER.debug("Won't merge with %s", other_predictable_field)
                continue

            LOGGER.debug("Would merge with %s", other_predictable_field)
            LOGGER.debug("Merged field: %s", merged_field)

            if _is_field_predictable(bodies, merged_field):
                merged_field = merged_field._replace(type=FieldType.MULTIPLEXER)

            # Merging a constant field doesn't affect predictability
            if merged_field.type in [ FieldType.CONST, FieldType.MULTIPLEXER ]:
                predictable_fields.remove(this_predictable_field)
                predictable_fields.remove(other_predictable_field)
                predictable_fields.insert(predictable_field_index, merged_field)
                break
        else:
            predictable_field_index += 1

    LOGGER.debug("-" * 100)
    return set(predictable_fields)


def _is_field_constant(bodies: np.ndarray, field: Field) -> bool:
    """
    Args:
        bodies: A (non-empty) NumPy array containing CAN packet bodies belonging to the same identifier. The
            `np.ndarray` contains elements of type `np.uint64`.
        field: The field to check constantness for.

    Returns:
        Whether the field is constant.
    """

    return (
        field.type is FieldType.CONST or
        (
            field.type is FieldType.UNKNOWN and
            len(np.unique(_extract_field(bodies, field))) == 1
        )
    )


def _extract_field(bodies: np.ndarray, field: Field) -> np.ndarray:
    """
    Args:
        bodies: A (non-empty) NumPy array containing CAN packet bodies belonging to the same identifier. The
            `np.ndarray` contains elements of type `np.uint64`.
        field: The field to extract.

    Returns:
        The extracted field values.
        The `np.ndarray` contains elements of type `np.uint64`.
    """

    # With DBC's "sawtooth" byte ordering and bit indexing behaviour, "little endian" byte order means that
    # semantically adjacent bits are also using adjacent bit indizes. I.e. the bit that semantically follows
    # bit 7 (counting from 0) is bit 8. For "bit endian", the indizes do not logically follow the semantic
    # ordering. For example, the semantically next bit after bit 15 is bit 0 in case of big endian. With
    # little endian, bits sawtooth into the _next_ byte, while for big endian, they sawtooth into the
    # _previous_ byte. These jumps make it hard to extract a semantically coherent big endian field, if it
    # crosses byte borders. The following code solves this problem by swapping the bytes of the CAN packet
    # body and updating the index of the starting bit of the field to extract. By swapping the bytes, fields
    # that used to sawtooth into the next byte will now sawtooth into the previous byte and vice versa, thus
    # big endian fields are then accessible conveniently like little endian fields.
    if field.endianness is FieldEndianness.BIG:
        # Swap the bytes to achieve convenient little endian-style access
        bodies = bodies.byteswap()

        # Update the anchors accordingly
        lsb_anchor = (7 - (field.lsb_anchor // 8)) * 8 + field.lsb_anchor % 8
        msb_anchor = (7 - (field.msb_anchor // 8)) * 8 + field.msb_anchor % 8

        # Update the field, so that the following code for little endian fields can be reused.
        field = Field(
            lsb_anchor=lsb_anchor,
            msb_anchor=msb_anchor,
            size=field.size,
            endianness=FieldEndianness.LITTLE,
            type=field.type
        )

    mask = 0xFFFFFFFFFFFFFFFF if field.size == 64 else ((1 << field.size) - 1)

    return (bodies >> field.lsb_anchor) & mask


def _is_field_predictable(bodies: np.ndarray, field: Field) -> bool:
    """
    Args:
        bodies: A (non-empty) NumPy array containing CAN packet bodies belonging to the same identifier. The
            `np.ndarray` contains elements of type `np.uint64`.
        field: The field to check predictability for.

    Returns:
        Whether the field seems to be predictable.
    """

    # Free parameters!
    training_sequence_length = 64
    max_num_training_samples = 8192

    def _seems_predictable(predicted: np.ndarray, actual: np.ndarray) -> bool:
        # Both `np.ndarray` contains elements of type `np.uint64`.

        # Count bit positions which have a TAV greater than 0, indicating that they are non-constant
        num_non_constant_bits: int = np.count_nonzero(_calculate_tav(actual, field.size) > 0)

        # Count entries where expected and actual differ by at least one bit. It doesn't matter by
        # how many bits the entries differ, either the prediction is correct or it is not.
        num_mistakes: int = np.count_nonzero(predicted ^ actual)

        LOGGER.debug(
            "Field %s, containing %s non-constant bit%s predicted with %s mistake%s",
            field,
            num_non_constant_bits,
            "" if num_non_constant_bits == 1 else "s",
            num_mistakes,
            "" if num_mistakes == 1 else "s"
        )

        # This one I'm very unsure about
        return num_mistakes <= num_non_constant_bits * 2

    if field.type is FieldType.MULTIPLEXER:
        return True

    if field.type is FieldType.UNKNOWN:
        # The bits corresponding to the field for each body
        field_values = _extract_field(bodies, field)

        LOGGER.debug("Checking predictablility of field %s.", field)

        # Prepare training samples of length training_sequence_length
        field_values_shifted = []
        for i in range(training_sequence_length):
            start = i
            end = field_values.shape[0] - training_sequence_length + i

            field_values_shifted.append(field_values[start:end])

        training_samples = np.stack(field_values_shifted, axis=-1)
        training_labels = field_values[training_sequence_length:]

        training_samples = training_samples[:max_num_training_samples]
        training_labels = training_labels[:max_num_training_samples]

        # Check whether the field is constant after skipping the first `delay` entries
        if len(np.unique(training_labels)) == 1:
            LOGGER.debug(
                "Field %s, containing 0 non-constant bits predicted with 0 single-bit mistakes",
                field
            )
            return True

        # If it is not constant, train a linear SVM and see whether it is capable of predicting bit
        # flips.
        classifier = svm.LinearSVC()
        try:
            classifier.fit(training_samples, training_labels)
        except ValueError:
            LOGGER.exception("Fitting the classifier for multiplexer field detection raised.")
            return False

        return _seems_predictable(classifier.predict(training_samples), training_labels)

    return False


def _merge_fields(a: Field, b: Field) -> Optional[Field]:
    """
    Try to merge two fields, based on their field positions and types.

    Args:
        a, b: The fields to merge.

    Returns:
        The merged field or 'None', if the fields could not be merged.

    Note:
        Field merging happens only based on the position and types of the fields. This function does not check
        whether the fields semantically merge.
    """

    # Merge the types:
    merged_type: Optional[FieldType] = None

    # Constant fields can be merged with any other type. To make type merging easier, swap a and b if b is
    # constant.
    if b.type is FieldType.CONST:
        a, b = b, a

    # Constant fields can be merged with any other type without losing semantics.
    if a.type is FieldType.CONST:
        merged_type = b.type

    # Two fields of type multiplexer or value can be merged, but semantics are potentially lost, thus the type
    # is reduced to unknown.
    if a.type is b.type and a.type in [ FieldType.MULTIPLEXER, FieldType.VALUE ]:
        merged_type = FieldType.UNKNOWN

    # If a merged type was not found at this point, abort.
    if merged_type is None:
        return None

    # Merge the size:
    merged_size = a.size + b.size

    # Merge anchors and endianness:
    merged_lsb_anchor = None
    merged_msb_anchor = None
    merged_endianness = None

    # Check which bytes are affected by the fields
    affected_bytes_a = _get_affected_bytes(a)
    affected_bytes_b = _get_affected_bytes(b)
    affected_bytes_both = affected_bytes_a & affected_bytes_b
    affected_bytes_any = affected_bytes_a | affected_bytes_b

    # Fields may have at most one affected byte in common, otherwise they are guaranteed to overlap.
    if len(affected_bytes_both) > 1:
        return None

    # If no common byte is affected by both fields, the LSB of one must be the byte after the MSB of the
    # other.
    if len(affected_bytes_both) == 0:
        b_after_a = max(affected_bytes_a) + 1 == min(affected_bytes_b)
        a_after_b = max(affected_bytes_b) + 1 == min(affected_bytes_a)

    # If a common byte is affected by both fields, it must be the MSB of one and the LSB of the other.
    if len(affected_bytes_both) == 1:
        b_after_a = max(affected_bytes_a) == min(affected_bytes_b)
        a_after_b = max(affected_bytes_b) == min(affected_bytes_a)

    # Check whether the affected bytes follow the above rules, to rule out a byte-level overlap.
    if not (b_after_a or a_after_b):
        return None

    # Swap the variables so that b follows a.
    if a_after_b:
        affected_bytes_a, affected_bytes_b = affected_bytes_b, affected_bytes_a
        a, b = b, a

        # Not used after this point but better safe than sorry
        b_after_a, a_after_b = a_after_b, b_after_a

    # The next step is to rule out a bit-level overlap and to make sure that the fields are adjacent on the
    # bit-level too:
    # Check which bits are affected by a and b at the (potential) border between them
    affected_border_bits_a = _get_affected_bits(a, max(affected_bytes_a))
    affected_border_bits_b = _get_affected_bits(b, min(affected_bytes_b))

    # This is where endianness comes into play: unknown endianness can be merged with any other endianness,
    # while big can not be merged with little.
    current_endianness = { a.endianness, b.endianness }

    # Check whether a merged field with unknown endianness can be created:
    # - Both fields must be of unknown endianness
    # - Both fields must affect the same byte
    # - No other bytes must be affected (theoretically implied by being unknown in the first place)
    # - The affected bits must not overlap
    # - The affected bits must be adjacent
    if (
        current_endianness == { FieldEndianness.UNKNOWN } and
        len(affected_bytes_both) == 1 and
        len(affected_bytes_any) == 1 and
        len(affected_border_bits_a & affected_border_bits_b) == 0
    ):
        if max(affected_border_bits_a) + 1 == min(affected_border_bits_b):
            # The fields are adjacent and of unknown endianness; b follows a
            merged_lsb_anchor = a.lsb_anchor
            merged_msb_anchor = b.msb_anchor
            merged_endianness = FieldEndianness.UNKNOWN

        if max(affected_border_bits_b) + 1 == min(affected_border_bits_a):
            # The fields are adjacent and of unknown endianness; a follows b
            merged_lsb_anchor = b.lsb_anchor
            merged_msb_anchor = a.msb_anchor
            merged_endianness = FieldEndianness.UNKNOWN

    # Check whether a merged field with little endianness can be created:
    # - Both fields must be of unknown or little endianness
    # - Multiple bytes must be affected
    # - In case there is no commonly affected byte:
    #   - Bit 7 of the MSB of a must be affected
    #   - Bit 0 of the LSB of b must be affected
    # - In case there is a commonly affected byte:
    #   - The affected bits must not overlap
    #   - The most significant bit affected by a must be adjacent to the least significant bit affected by b
    if (
        current_endianness <= { FieldEndianness.LITTLE, FieldEndianness.UNKNOWN } and
        len(affected_bytes_any) > 1 and
        (
            (
                len(affected_bytes_both) == 0 and
                7 in affected_border_bits_a and
                0 in affected_border_bits_b
            ) or
            (
                len(affected_bytes_both) == 1 and
                len(affected_border_bits_a & affected_border_bits_b) == 0 and
                max(affected_border_bits_a) + 1 == min(affected_border_bits_b)
            )
        )
    ):
        merged_lsb_anchor = a.lsb_anchor
        merged_msb_anchor = b.msb_anchor
        merged_endianness = FieldEndianness.LITTLE

    # Check whether a merged field with big endianness can be created:
    # - Both fields must be of unknown or big endianness
    # - Multiple bytes must be affected
    # - In case there is no commonly affected byte:
    #   - Bit 0 of the MSB of a must be affected
    #   - Bit 7 of the LSB of b must be affected
    # - In case there is a commonly affected byte:
    #   - The affected bits must not overlap
    #   - The most significant bit affected by b must be adjacent to the least significant bit affected by a
    if (
        current_endianness <= { FieldEndianness.BIG, FieldEndianness.UNKNOWN } and
        len(affected_bytes_any) > 1 and
        (
            (
                len(affected_bytes_both) == 0 and
                0 in affected_border_bits_a and
                7 in affected_border_bits_b
            ) or
            (
                len(affected_bytes_both) == 1 and
                len(affected_border_bits_a & affected_border_bits_b) == 0 and
                max(affected_border_bits_b) + 1 == min(affected_border_bits_a)
            )
        )
    ):
        merged_lsb_anchor = b.lsb_anchor
        merged_msb_anchor = a.msb_anchor
        merged_endianness = FieldEndianness.BIG

    # Make sure that all properties could be merged.
    if (
        merged_lsb_anchor is None or
        merged_msb_anchor is None or
        merged_size is None or
        merged_endianness is None or
        merged_type is None
    ):
        return None

    return Field(
        lsb_anchor=merged_lsb_anchor,
        msb_anchor=merged_msb_anchor,
        size=merged_size,
        endianness=merged_endianness,
        type=merged_type
    )


def _get_affected_bytes(field: Field) -> Set[int]:
    """
    Args:
        field: A field.

    Returns:
        The indices of all bytes affected by this field.
    """

    lsb = field.lsb_anchor // 8
    msb = field.msb_anchor // 8

    return set(range(min(lsb, msb), max(lsb, msb) + 1))


def _get_affected_bits(field: Field, byte: int) -> Set[int]:
    """
    Args:
        field: A field.
        byte: A byte index.

    Returns:
        The indices of all bits affected by the byte of this field.
    """

    if byte not in _get_affected_bytes(field):
        return set()

    if field.endianness is FieldEndianness.BIG:
        # "Convert" the big endian field into a little endian field.
        byte = 2 * (field.lsb_anchor // 8) - byte

        field = field._replace(msb_anchor=(field.lsb_anchor + field.size - 1))

    byte_lsb_anchor = byte * 8
    byte_msb_anchor = ((byte + 1) * 8) - 1

    affected_relative_lsb_anchor = max(byte_lsb_anchor, field.lsb_anchor) % 8
    affected_relative_msb_anchor = min(byte_msb_anchor, field.msb_anchor) % 8

    return set(range(affected_relative_lsb_anchor, affected_relative_msb_anchor + 1))


def _demux_by(bodies: np.ndarray, field: Field) -> Dict[np.uint64, np.ndarray]:
    """
    Args:
        bodies: A (non-empty) NumPy array containing CAN packet bodies belonging to the same identifier. The
            `np.ndarray` contains elements of type `np.uint64`.
        field: The field to demultiplex by.

    Returns:
        For each value the multiplexer field takes, a list of packet bodies.
        The `np.ndarray` contains elements of type `np.uint64`.

    Raises:
        OverflowError: if there are too many unique values of the multiplexer field.
    """

    # Free parameters!
    max_unique_mux_values = 2 ** 8

    if field.type is not FieldType.MULTIPLEXER:
        raise ValueError("Not a multiplexer field.")

    field_values = _extract_field(bodies, field)

    # The bits corresponding to the multiplexer field for each body
    unique_mux_values, inverse_body_indizes = np.unique(field_values, return_inverse=True)
    if unique_mux_values.shape[0] > max_unique_mux_values:
        raise OverflowError("Too many unique values of the multiplexer field ({}).".format(
            unique_mux_values.shape[0]
        ))

    return { v: bodies[inverse_body_indizes == i] for i, v in enumerate(unique_mux_values) }


class DemultiplexedAnalysisResult(NamedTuple):
    tav: np.ndarray  # Containing np.uint64
    relative_tav: np.ndarray  # Containing np.float64
    tav_derivative: np.ndarray  # Containing np.float64
    bcot: np.ndarray  # Containing np.float64
    fields: Set[Field]


def analyze_demultiplexed(bodies: np.ndarray, size: int) -> DemultiplexedAnalysisResult:
    """
    Find field boundaries in CAN packet bodies. This assumes that all bodies belong to the same CAN identifier
    and that multiplexers have already been detected and the bodies were demultiplexed.
    Do NOT strip the multiplexer field(s) off the bodies before passing them here. That is, pass all bodies
    that have the same value for all multiplexer fields. The multiplexer fields will then be detected as
    constants.

    Args:
        bodies: The bodies to analyze. The `np.ndarray` contains elements of type `np.uint64`.
        size: The number of bits in each body. All bodies must have the same bit size.

    Returns:
        The results of this analysis.

    Note:
        Always pass the full CAN bodies. Failing to do so messes with bit and byte positions
        required for little/big endian detection and field merging.

    Raises:
        NotEnoughData: if the data is too sparse to perform analysis.
    """

    # Free parameters!
    if size < 1:
        raise ValueError("Bodies must consist of at least one bit.")

    if len(bodies) < 2:
        raise NotEnoughData("Need at least two bodies to perform any analysis.")

    # Prepare all metrics
    tav = _calculate_tav(bodies, size)
    relative_tav = _calculate_relative_tav(tav)
    tav_derivative = _calculate_tav_derivative(relative_tav)
    bcot = _calculate_bcot(bodies, size)

    # Roughly detect fields
    rough_field_separators = _find_rough_field_separators(tav_derivative, bcot)

    # Add all byte borders as potential field separators
    rough_field_separators |= set(x for x in [7, 15, 23, 31, 39, 47, 55, 63] if x < size)
    rough_field_separators_ascending = sorted(list(rough_field_separators))

    LOGGER.debug("Rough field separators: %s", rough_field_separators_ascending)

    # Use the rough field separators to classify first CONST and VALUE fields
    fields = []
    field_start = 0
    for separator in rough_field_separators_ascending:
        field_end = separator
        field_size = field_end - field_start + 1

        field = Field(
            lsb_anchor=field_start,
            msb_anchor=field_end,
            size=field_size,
            endianness=FieldEndianness.UNKNOWN,
            type=FieldType.UNKNOWN
        )

        if _is_field_constant(bodies, field):
            field = field._replace(type=FieldType.CONST)
        else:
            field = field._replace(type=FieldType.VALUE)

        # This search finds little endian or unknown endian fields.
        if len(_get_affected_bytes(field)) > 1:
            field = field._replace(endianness=FieldEndianness.LITTLE)

        fields.append(field)

        field_start = separator + 1

    # Try to merge fields to build larger ones
    field_index = 0
    while field_index < len(fields) - 1:
        this_field = fields[field_index]

        LOGGER.debug("\t" + "-" * 100)  # pylint: disable=logging-not-lazy
        LOGGER.debug("\tFinding a merging buddy for %s", this_field)

        for other_field in fields[field_index + 1:]:
            merged_field = _merge_fields(this_field, other_field)
            if merged_field is None:
                LOGGER.debug("\tWon't merge with %s", other_field)
                continue

            LOGGER.debug("\tWould merge with %s", other_field)

            merged_field = _restore_merge_semantics(
                bodies,
                merged_field,
                this_field,
                other_field
            )

            if merged_field is not None:
                fields.remove(this_field)
                fields.remove(other_field)
                fields.insert(field_index, merged_field)
                break
        else:
            field_index += 1

    return DemultiplexedAnalysisResult(
        tav=tav,
        relative_tav=relative_tav,
        tav_derivative=tav_derivative,
        bcot=bcot,
        fields=set(fields)
    )


def _calculate_tav(bodies: np.ndarray, size: int) -> np.ndarray:
    """
    Args:
        bodies: The bodies to analyze. The `np.ndarray` contains elements of type `np.uint64`.
        size: The number of bits in each body. All bodies must have the same bit size.

    Returns:
        The absolute TAV, i.e. for each bit position the absolute number of bit flips.
        The `np.ndarray` contains elements of type `np.uint64`.
    """

    if size < 1:
        raise ValueError("Bodies must consist of at least one bit.")

    tav = np.zeros(size, dtype=np.uint64)
    for bit in np.arange(size):
        bits = (bodies >> bit) & 1
        tav[bit] = np.sum(bits[1:] ^ bits[:-1])
    return tav


def _calculate_relative_tav(tav: np.ndarray) -> np.ndarray:
    """
    Args:
        tav: The (absolute) TAV, as returned by `_calculate_tav`. The `np.ndarray` contains elements of type
            `np.uint64`.

    Returns:
        The relative TAV, i.e. for each bit position the number of bit flips in relation to all other bit
        flips. The relative TAV for all bit positions adds up to 1.
        The `np.ndarray` contains elements of type `np.float64`.
    """

    tav = tav.astype(np.float64)

    return tav / np.linalg.norm(tav)


def _calculate_tav_derivative(relative_tav: np.ndarray) -> np.ndarray:
    """
    Args:
        relative_tav: The relative TAV, as returned by `_calculate_relative_tav`. The `np.ndarray` contains
            elements of type `np.float64`.

    Returns:
        The derivative of the relative TAV. Relates adjacent bit positions, thus the entry "0" belongs to the
        relation between bit positions 0 and 1.
        The `np.ndarray` contains elements of type `np.float64`.
    """

    if relative_tav.shape[0] < 1:
        raise ValueError("The TAV must be available for at least one bit.")

    return relative_tav[1:] - relative_tav[:-1]


def _calculate_bcot(bodies: np.ndarray, size: int) -> np.ndarray:
    """
    Args:
        bodies: The bodies to analyze. The `np.ndarray` contains elements of type `np.uint64`.
        size: The number of bits in each body. All bodies must have the same bit size.

    Returns:
        The Bit-Correlation-Over-Time. Like the derivative of the TAV, this metric relates adjacent bit
        positions, thus the entry "0" belongs to the relation between bit positions 0 and 1. Note that entries
        might be nan (= not a number), in case at least one of the correlated bits is constant. For example,
        if bit 4 is constant, the entries "3" and "4" will be nan, because the correlation with a constant bit
        is undefined.
        The `np.ndarray` contains elements of type `np.float64`.
    """

    # Free parameters!
    bcot_max_samples = 64 * 1024
    convolution_length = max(min(bodies.shape[0], bcot_max_samples) // 200, 64)

    if size < 1:
        raise ValueError("Bodies must consist of at least one bit.")

    bodies = bodies[:bcot_max_samples]

    # Note: this code works with temporary Python list, which are potential bottlenecks, but the
    # lists only have one entry per bit position (minus one), so the worst case is 63 entries per
    # list, which should not be an issue.
    # Note: Variable names are chosen as per the paper that defines this algorithm.
    b = bodies[1:] ^ bodies[:-1]  # pylint: disable=invalid-name

    b_t = np.array([ ((b >> col) & 1) for col in np.arange(size) ], dtype=np.uint8)
    v_t = np.ones((size, convolution_length), dtype=np.uint8)
    c_t = np.array([ np.convolve(b_t[row], v_t[row]) for row in np.arange(size) ])
    bcot = np.array([ np.corrcoef(c_t[row], c_t[row + 1])[1][0] for row in np.arange(size - 1) ])

    return bcot.astype(np.float64)


def _find_rough_field_separators(
    tav_derivative: np.ndarray,
    bcot: np.ndarray
) -> Set[int]:
    """
    Args:
        tav_derivative: The derivative of the (relative) TAV, as returned by `_calculate_tav_derivative`. The
            `np.ndarray` contains elements of type `np.float64`.
        bcot: The Bit-Correlation-Over-Time, as returned by `_calculate_bcot`. The `np.ndarray` contains
            elements of type `np.float64`.

    Returns:
        The (rough) field separators detected in the CAN packet bodies. Here "5" means that one field ends at
        bit position 5 and another field starts at bit position 6.

    Raises:
        NotEnoughData: if the data is too sparse to perform rough field separation.
    """

    # Free parameters!
    num_classes = 2
    num_separator_classes = 1
    min_dist = .55

    tav_derivative_vs_bcot = np.stack((bcot, tav_derivative), axis=-1)
    non_nan_condition = np.logical_not(np.isnan(tav_derivative_vs_bcot[:, 0]))
    non_nan_indizes = np.arange(bcot.shape[0])[non_nan_condition]
    point_cloud = tav_derivative_vs_bcot[non_nan_condition]

    if len(point_cloud) == 0 or len(np.unique(point_cloud, axis=0)) < num_classes:
        raise NotEnoughData("Not enough unique points to separate {} classes.".format(num_classes))

    dist = minkowski(np.array(np.min(point_cloud[:, 0]), np.min(point_cloud[:, 1])),
                     np.array(np.max(point_cloud[:, 0]), np.max(point_cloud[:, 1])))

    if dist < min_dist:
        LOGGER.debug("All points are close, assuming no separators.")
        return set()

    kmeans = cluster.KMeans(n_clusters=num_classes).fit(point_cloud)

    separator_classes = list(map(lambda x: x[0], sorted(
        enumerate(kmeans.cluster_centers_),
        key=lambda x: cast(float, np.linalg.norm(x[1]))
    )[:num_separator_classes]))

    separators = []
    for separator_class in separator_classes:
        separators.extend(non_nan_indizes[kmeans.labels_ == separator_class])

    # Return the separating bit positions
    return { int(x) for x in separators }


def _restore_merge_semantics(
    bodies: np.ndarray,
    merged_field: Field,
    a: Field,
    b: Field
) -> Optional[Field]:
    """
    Args:
        bodies: The bodies which are subject to analysis. The `np.ndarray` contains elements of type
            `np.uint64`.
        merged_field: The merged field to restore semantics for.
        a: The first source field of the merged field.
        b: The second source field of the merged field.

    Returns:
        The merged field with restored semantics if possible, 'None' otherwise.
    """

    if merged_field.type is not FieldType.UNKNOWN:
        return merged_field

    if not (a.type is b.type is FieldType.VALUE):  # pylint: disable=superfluous-parens
        return None

    # Extract the field bodies
    merged_field_bodies = _extract_field(bodies, merged_field)

    # Prepare all metrics
    tav = _calculate_tav(merged_field_bodies, merged_field.size)
    relative_tav = _calculate_relative_tav(tav)
    tav_derivative = _calculate_tav_derivative(relative_tav)
    bcot = _calculate_bcot(merged_field_bodies, merged_field.size)

    # Roughly detect fields
    try:
        if len(_find_rough_field_separators(tav_derivative, bcot)) == 0:
            return merged_field._replace(type=FieldType.VALUE)
    except NotEnoughData:
        # Not sure whether merging here is the correct decision, but it reduces single-bit fields.
        return merged_field._replace(type=FieldType.VALUE)

    return None


def _solve_overlaps(
    analysis_result: DemultiplexedAnalysisResult,
    field: Field,
    size: int
) -> DemultiplexedAnalysisResult:
    """
    Adds `field` to the analysis result, resizing analysed fields as required to avoid overlaps.

    Args:
        analysis_result: The result of an analysis.
        field: The field to add/fit into the analysis result.
        size: The size of the CAN message in bytes.

    Returns:
        The updated analysis result including `field`.
    """

    LOGGER.debug("Fitting %s into %s", field, analysis_result)

    for analyzed_field in set(analysis_result.fields):
        analysis_result.fields.remove(analyzed_field)
        analysis_result.fields.update(_solve_overlap(analyzed_field, field, size))

    analysis_result.fields.add(field)

    return analysis_result


def _solve_overlap(analyzed_field: Field, field: Field, size: int) -> Set[Field]:
    """
    Args:
        analyzed_field: Field to modify to avoid overlaps with `field`.
        field: The field to avoid overlaps with.
        size: The size of the CAN message in bytes.

    Returns:
        `analyzed_field`, modified/split/removed to avoid overlap with `field`.
    """

    # Resolving overlaps between fields is rather complicated, especially given the different
    # possible combinations of endianness. The following code works around this by looking at which
    # bits are affected by each field, "masking" one field off of the other and then merging the
    # remaining bits back into fields.
    masked_analyzed_field_bits = {
        # Take the bits affected by analyzed_field and remove those affected by field.
        # This effectively performs the bitwise operation 'analyzed_field & ~field'
        byte: _get_affected_bits(analyzed_field, byte) & (set(range(8)) - _get_affected_bits(field, byte))
        for byte in range(size)
    }

    # Restore field structures from the remaining bits
    masked_bytewise_fields = []

    # Find adjacent bits and build (small) fields for those ranges
    for byte, bits in masked_analyzed_field_bits.items():
        bytewise_field = None

        for bit in range(8):
            if bit in bits:
                if bytewise_field is None:
                    bytewise_field = Field(
                        lsb_anchor=(byte * 8 + bit),
                        msb_anchor=(byte * 8 + bit),
                        size=1,
                        endianness=analyzed_field.endianness,
                        type=analyzed_field.type
                    )

                bytewise_field = bytewise_field._replace(msb_anchor=(byte * 8 + bit))
                bytewise_field = bytewise_field._replace(
                    size=(bytewise_field.msb_anchor - bytewise_field.lsb_anchor + 1)
                )
            else:
                if bytewise_field is not None:
                    masked_bytewise_fields.append(bytewise_field)
                bytewise_field = None

        if bytewise_field is not None:
            masked_bytewise_fields.append(bytewise_field)

    # Merge the (small) bytewise fields to build the final non-overlapping field(s)
    field_index = 0
    while field_index < len(masked_bytewise_fields) - 1:
        this_field = masked_bytewise_fields[field_index]

        for other_field in masked_bytewise_fields[field_index + 1:]:
            merged_field = _merge_fields(this_field, other_field)
            if merged_field is None:
                continue

            masked_bytewise_fields.remove(this_field)
            masked_bytewise_fields.remove(other_field)
            masked_bytewise_fields.insert(field_index, merged_field._replace(
                type=analyzed_field.type
            ))
            break
        else:
            field_index += 1

    LOGGER.debug("Avoiding overlaps between %s and %s: %s", analyzed_field, field, masked_bytewise_fields)

    return set(masked_bytewise_fields)


def _restore_dbc(
    identifier: int,
    size: int,
    top_level_fields: Set[Field],
    mux_level_fields: Optional[Dict[int, Set[Field]]],
    output_directory: str,
    suffix: Optional[str] = None
) -> Tuple[str, cantools.db.Database]:
    """
    Args:
        identifier: The identifier of this CAN message.
        size: The size of the CAN message in bytes.
        top_level_fields: Fields detected on the message top level (i.e. not multiplexed).
        mux_level_fields: ...
        output_directory: ...
        suffix: Suffix to append to the file name (but in front of the extension).

    Returns:
        The path pointing to the restored DBC file and the loaded DBC.
    """

    signals: List[cantools.db.Signal] = []

    multiplexer_signal_name = None
    for field in top_level_fields:
        start = field.msb_anchor if field.endianness is FieldEndianness.BIG else field.lsb_anchor

        name = "TOP_LEVEL_{}_{}_{}".format(
            start,
            field.size,
            field.endianness.to_cantools()
        )

        if field.type is FieldType.MULTIPLEXER:
            multiplexer_signal_name = name

        signals.append(cantools.database.can.Signal(
            name=name,
            start=start,
            length=field.size,
            byte_order=field.endianness.to_cantools(),
            is_multiplexer=field.type is FieldType.MULTIPLEXER,
            is_float=False  # TODO
        ))

    if mux_level_fields is not None:
        for mux_value, fields in mux_level_fields.items():
            for field in fields:
                start = field.msb_anchor if field.endianness is FieldEndianness.BIG else field.lsb_anchor

                name = "MUX_{}_LEVEL_{}_{}_{}".format(
                    mux_value,
                    start,
                    field.size,
                    field.endianness.to_cantools()
                )

                signals.append(cantools.database.can.Signal(
                    name=name,
                    start=start,
                    length=field.size,
                    byte_order=field.endianness.to_cantools(),
                    multiplexer_ids=[ int(mux_value) ],
                    multiplexer_signal=multiplexer_signal_name,
                    is_float=False  # TODO
                ))

    dbc_output = os.path.join(output_directory, "restored{}.dbc".format(suffix or ""))
    LOGGER.info("DBC output: %s", dbc_output)

    try:
        db = cantools.database.load_file(dbc_output)
        LOGGER.info("Extending existing DBC.")
    except (IOError, OSError):
        db = cantools.database.can.Database()
        LOGGER.info("Creating new DBC.")

    db.messages.append(cantools.database.can.Message(
        identifier,
        "ID{}".format(identifier),
        size,
        signals
    ))
    db.refresh()

    cantools.database.dump_file(db, dbc_output)

    return (dbc_output, db)

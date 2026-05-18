from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WheeltecPacket:
    """
    Wheeltec WiFi control packet used by the provided reference scripts:

      payload = 0xAA + float32(u_L) + float32(u_R) + checksum(uint8)

    checksum is the sum of the first 9 bytes modulo 256.
    """

    u_l: float
    u_r: float

    def encode(self) -> bytes:
        pkt = struct.pack("<Bff", 0xAA, float(self.u_l), float(self.u_r))
        checksum = sum(pkt) & 0xFF
        return pkt + bytes([checksum])


def decode_state_frame(text: str) -> Optional[Tuple[float, ...]]:
    """
    Decode a state frame from the robot.

    The provided wifi3.0.py expects frames like:
      {...} with floats separated by ':'
    and checks len(values) == 13.

    This function extracts the first {...} frame from a buffer chunk.
    Returns the tuple of floats if successful, else None.
    """
    start = text.find("{")
    if start == -1:
        return None
    end = text.find("}", start)
    if end == -1:
        return None
    frame = text[start + 1 : end]
    try:
        values = tuple(float(x) for x in frame.split(":"))
    except ValueError:
        return None
    if len(values) != 13:
        return None
    return values


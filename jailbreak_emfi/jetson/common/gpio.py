"""One GPIO output line (libgpiod, v1 or v2 Python bindings), driven high while the glitched GPU work runs
(a marker for the oscilloscope; the glitch itself is timed from the near-field probe edge)."""

from __future__ import annotations


class GpioOutput:
    def __init__(self, chip_name: str, line_number: int, consumer: str):
        self.chip_name, self.line_number, self.consumer = chip_name, int(line_number), consumer
        self._gpiod = self._chip = self._line = self._request = None

    def __enter__(self) -> "GpioOutput":
        import gpiod

        self._gpiod = gpiod
        try:
            if hasattr(gpiod, "request_lines"):          # libgpiod v2
                settings = {self.line_number: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT,
                                                                 output_value=gpiod.line.Value.INACTIVE)}
                self._request = gpiod.request_lines(self.chip_name, consumer=self.consumer, config=settings)
            else:                                        # libgpiod v1
                self._chip = gpiod.Chip(self.chip_name)
                self._line = self._chip.get_line(self.line_number)
                self._line.request(consumer=self.consumer, type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
            self.set_value(0)
            return self
        except BaseException:
            self.close()
            raise

    def set_value(self, value: int) -> None:
        if self._request is not None:
            level = self._gpiod.line.Value.ACTIVE if value else self._gpiod.line.Value.INACTIVE
            self._request.set_value(self.line_number, level)
        elif self._line is not None:
            self._line.set_value(1 if value else 0)

    def close(self) -> None:
        try:
            self.set_value(0)
        finally:
            if self._request is not None:
                self._request.release()
                self._request = None
            if self._line is not None:
                self._line.release()
                self._line = None
            if self._chip is not None:
                self._chip.close()
                self._chip = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

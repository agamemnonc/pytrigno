This is the Intelligent Sensing fork of  `pytrigno <https://github.com/axopy/pytrigno>`_.

pytrigno
========

``TrignoEMG`` and ``TrignoAccel`` provide access to data served by Trigno
Control Utility for the Delsys Trigno wireless EMG system. TCU is Windows-only,
but this class can be used to stream data from it on another machine. TCU works
by running a TCP/IP server, with EMG data from the sensors on one port,
accelerometer data on another, and commands/responses on yet another. These
ports are configurable in the TCU GUI. The TCU program must be running before
a ``TrignoEMG`` or ``TrignoAccel`` object is created.

EMG data is sampled at 2000 Hz and is in volts (by default) with a range of
±0.011 V. This can be converted to millivolts or normalized by the max range to
get a range of ±11 mV or ±1 (unitless), respectively.

Accelerometer data is sampled at 148.1 Hz and is in g.

You can test operation of the device by running ``examples/check_trigno.py`` to
see if things are set up correctly -- if no errors occur, it should be ready to
go.

Tested with Delsys Trigno Control Utility v. 3.5.1. The software can be downloaded from the `Delsys support site <https://www.delsys.com/support/software/>`_.

Dependencies
------------

- `NumPy <http://www.numpy.org/>`_

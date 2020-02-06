from abc import ABC
import socket
import struct
import warnings

import numpy as np


class _BaseTrignoDaq(ABC):
    """
    Delsys Trigno base class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    BYTES_PER_CHANNEL = 4
    TOTAL_NUM_CHANNELS = 16
    CMD_TERM = '\r\n\r\n'

    def __init__(
            self,
            host,
            cmd_port,
            data_port,
            timeout):
        self.host = host
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.timeout = timeout

        self.total_signals = \
            self._signals_per_channel * self.TOTAL_NUM_CHANNELS
        self._min_recv_size = self.total_signals * self.BYTES_PER_CHANNEL

        self._initialize()

    def _initialize(self):
        # create command socket and consume the servers initial response
        self._comm_socket = socket.create_connection(
            (self.host, self.cmd_port), self.timeout)
        self._comm_socket.recv(1024)

        # create the data socket
        self._data_socket = socket.create_connection(
            (self.host, self.data_port), self.timeout)

    def start(self):
        """
        Tell the device to begin streaming data.

        You should call ``read()`` soon after this, though the device typically
        takes about two seconds to send back the first batch of data.
        """
        resp = self._send_cmd('START')
        self._validate(resp)

    def set_channels(self, channels, zero_based):
        """
        Sets the channels to read from the device.

        Parameters
        ----------
        channels : list or tuple
            Sensor channels to use.
        """
        self.channels = list(set(channels))

        if zero_based:
            channels_ = self.channels
        else:
            channels_ = [channel - 1 for channel in self.channels]

        read_idx = np.zeros(0, dtype=int)
        for channel in channels_:
            read_idx = np.append(read_idx, np.arange(
                channel * self._signals_per_channel,
                channel * self._signals_per_channel +
                self._relevant_signals_per_channel))

        self._signals_read_idx = read_idx

    def read(self, num_samples):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Parameters
        ----------
        num_samples : int
            Number of samples to read per channel.

        Returns
        -------
        data : ndarray, shape=(total_signals, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        l_des = num_samples * self._min_recv_size
        l = 0
        packet = bytes()
        while l < l_des:
            try:
                packet += self._data_socket.recv(l_des - l)
            except socket.timeout:
                l = len(packet)
                packet += b'\x00' * (l_des - l)
                raise IOError("Device disconnected.")
            l = len(packet)

        data = np.asarray(
            struct.unpack(
                '<' +
                'f' *
                self.total_signals *
                num_samples,
                packet))
        data = np.transpose(data.reshape((-1, self.total_signals)))

        return data[self._signals_read_idx, :]

    def get_mode(self, sensor):
        """Queries operation mode for the specified sensor.

        Parameters
        ----------
        sensor : int
            Sensor number.

        Returns
        -------
        mode : int
            Operation mode.
        """
        response = self._send_cmd("SENSOR " + str(sensor) + " MODE?")
        mode = int(response)

        return mode

    def set_mode(self, sensor, mode):
        """Sets operation mode for the specified sensor.

        Parameters
        ----------
        sensor : int
            Sensor number.

        mode : int
            Operation mode.

        Returns
        -------
        response : str
            Server response message.
        """
        response = self._send_cmd(
            'SENSOR ' + str(sensor) + ' SETMODE ' + str(mode))

        return response

    def stop(self):
        """Tell the device to stop streaming data. """
        resp = self._send_cmd('STOP')
        self._validate(resp)

    def close_connection(self):
        """Closes the connection to the Trigno Control Utility server. """
        self._comm_socket.close()

    def reset(self):
        """Restart the connection to the Trigno Control Utility server. """
        self.close_connection()
        self._initialize()

    def __del__(self):
        try:
            self.close_connection()
        except BaseException:
            pass

    def _send_cmd(self, command):
        self._comm_socket.send(self._cmd(command))
        resp = self._comm_socket.recv(128)
        resp = self._process_response(resp)

        return resp

    @staticmethod
    def _cmd(command):
        return bytes("{}{}".format(command, _BaseTrignoDaq.CMD_TERM),
                     encoding='ascii')

    @staticmethod
    def _process_response(response):
        response = response.decode('utf-8')
        response = response[:-4]

        return response

    @staticmethod
    def _validate(response):
        if 'OK' not in response:
            warnings.warn(response)


class _BaseQuattro(ABC):
    """Delsys Quattro abstract class.

    Implements method for setting sensor operation mode.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def _set_sensor_modes(self, sensors, mode):
        """Sets channel mode for specified channels. """
        for sensor in sensors:
            if self.get_mode(sensor) != mode:
                self.set_mode(sensor, mode)


class _BaseTrignoEMGDaq(_BaseTrignoDaq):
    """
    Delsys Trigno EMG abstract class.

    Implements scaling of EMG data to convert into desired units.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
            self,
            samples_per_read,
            zero_based,
            units,
            host,
            cmd_port,
            data_port,
            timeout):
        super(_BaseTrignoEMGDaq, self).__init__(
            host=host, cmd_port=cmd_port, data_port=data_port, timeout=timeout)

        self.samples_per_read = samples_per_read
        self.zero_based = zero_based
        self.units = units

        if self.units == 'V':
            self.scaler_ = 1.
        elif self.units == 'mV':
            self.scaler_ = 1000.
        elif self.units == 'normalized':
            # max range of EMG data is 11 mV
            self.scaler_ = 1 / 0.011
        else:
            raise ValueError("Invalid unit {}.".format(self.units))

    def read(self):
        """
        Request a sample of data from the device.

        This is a blocking method, meaning it returns only once the requested
        number of samples are available.

        Returns
        -------
        data : ndarray, shape=(num_channels, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(_BaseTrignoEMGDaq, self).read(self.samples_per_read)
        return self.scaler_ * data


class TrignoEMG(_BaseTrignoEMGDaq):
    """
    Delsys Trigno EMG data acquisition.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has a single EMG
        channel.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    zero_based : boolean, optional (default: False)
        Whether channel numbering follows zero-based convention.
    units : {'V', 'mV', 'normalized'}, optional (default: 'V')
        Units in which to return data. If 'V', the data is returned in its
        un-scaled form (volts). If 'mV', the data is scaled to millivolt level.
        If 'normalized', the data is scaled by its maximum level so that its
        range is [-1, 1].
    host : str, optional (default: 'localhost')
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional (default: 50040)
        Port of TCU command messages.
    data_port : int, optional (default: 50043)
        Port of TCU EMG data access.
    timeout : float, optional (default: 5)
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    scaler : float
        Scaling factor used to convert the signals into the specified units.
    """
    _signals_per_channel = 1
    _relevant_signals_per_channel = 1

    def __init__(
            self,
            channels,
            samples_per_read,
            zero_based=False,
            units='V',
            host='localhost',
            cmd_port=50040,
            data_port=50043,
            timeout=5):
        super(TrignoEMG, self).__init__(
            samples_per_read=samples_per_read,
            zero_based=zero_based,
            units=units,
            host=host,
            cmd_port=cmd_port,
            data_port=data_port,
            timeout=timeout)

        self.set_channels(channels=channels, zero_based=zero_based)


class QuattroEMG(_BaseTrignoEMGDaq, _BaseQuattro):
    """
    Delsys Quattro EMG data acquisition.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has four EMG channels. Only include
        the paired slot numbers. For example, if using a single sensor paired
        at slot 1 set `channels=[1]`.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    zero_based : boolean, optional (default: False)
        Whether channel numbering follows zero-based convention.
    units : {'V', 'mV', 'normalized'}, optional (default: 'V')
        Units in which to return data. If 'V', the data is returned in its
        un-scaled form (volts). If 'mV', the data is scaled to millivolt level.
        If 'normalized', the data is scaled by its maximum level so that its
        range is [-1, 1].
    mode : int, optional (default: 313)
        Operation sensor mode.
    host : str, optional (default: 'localhost')
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional (default: 50040)
        Port of TCU command messages.
    data_port : int, optional (default: 50043)
        Port of TCU EMG data access.
    timeout : float, optional (default: 5)
        Number of seconds before socket returns a timeout exception.

    Attributes
    ----------
    scaler : float
        Scaling factor used to convert the signals into the specified units.
    """
    _signals_per_channel = 1
    _relevant_signals_per_channel = 1

    def __init__(
            self,
            sensors,
            samples_per_read,
            zero_based=False,
            units='V',
            mode=313,
            host='localhost',
            cmd_port=50040,
            data_port=50043,
            timeout=5):
        super(QuattroEMG, self).__init__(
            samples_per_read=samples_per_read,
            zero_based=zero_based,
            units=units,
            host=host,
            cmd_port=cmd_port,
            data_port=data_port,
            timeout=timeout)

        self.sensors = sensors
        self.mode = mode

        channels = self._channels_from_sensors(
            sensors=sensors,
            zero_based=zero_based)

        self.set_channels(channels=channels, zero_based=zero_based)
        self._set_sensor_modes(sensors=sensors, mode=mode)

    def _channels_from_sensors(self, sensors, zero_based):
        """Maps sensor (i.e. slot) numbers to channel numbers."""
        if zero_based:
            sensors = [sensor + 1 for sensor in sensors]

        channels = []
        for sensor in sensors:
            start = self._send_cmd("SENSOR " + str(sensor) + " STARTINDEX?")
            [channels.append(chan) for chan in list(
                range(int(start), int(start) + 4))]

        if zero_based:
            channels = [channel - 1 for channel in channels]

        return channels


class _BaseTrignoAuxDaq(_BaseTrignoDaq):
    """Delsys Trigno Auxiliary base class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def __init__(
            self,
            samples_per_read,
            zero_based,
            host,
            cmd_port,
            data_port,
            timeout):
        super(_BaseTrignoAuxDaq, self).__init__(
            host=host,
            cmd_port=cmd_port,
            data_port=data_port,
            timeout=timeout)

        self.samples_per_read = samples_per_read
        self.zero_based = zero_based

    def read(self):
        """
        Request a sample of data from the device.
        This is a blocking method, meaning it returns only once the requested
        number of samples are available.
        Returns
        -------
        data : ndarray, shape=(num_channels * 3, num_samples)
            Data read from the device. Each channel is a row and each column
            is a point in time.
        """
        data = super(_BaseTrignoAuxDaq, self).read(self.samples_per_read)
        return data


class TrignoACC(_BaseTrignoAuxDaq):
    """
    Delsys Trigno Acc data acquisition.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has three Acc channels.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    zero_based : boolean, optional (default: False)
        Whether channel numbering follows zero-based convention.
    host : str, optional (default: 'localhost')
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional (default: 50040)
        Port of TCU command messages.
    data_port : int, optional (default: 50042)
        Port of TCU EMG data access.
    timeout : float, optional (default: 5)
        Number of seconds before socket returns a timeout exception.

    Notes
    -----
    This class should only be used for the legacy Trigno sensors (EMG & Acc).
    For reading acceleration data from Trigno IM sensors, use the ``TrignoIMU``
    class instead followed by appropriate indexing.
    """
    _signals_per_channel = 3
    _relevant_signals_per_channel = 3

    def __init__(
            self,
            channels,
            samples_per_read,
            zero_based=False,
            host='localhost',
            cmd_port=50040,
            data_port=50044,
            timeout=5):
        super(TrignoACC, self).__init__(
            samples_per_read=samples_per_read,
            zero_based=zero_based,
            host=host,
            cmd_port=cmd_port,
            data_port=data_port,
            timeout=timeout)

        self.set_channels(channels=channels, zero_based=zero_based)


class TrignoIMU(_BaseTrignoAuxDaq):
    """
    Delsys Trigno IMU data acquisition.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has three, four or nine IMU
        channels, depending on `imu_type`.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    zero_based : boolean, optional (default: False)
        Whether channel numbering follows zero-based convention.
    imu_mode : {'raw', 'quaternion', 'euler'}, optional (default: 'raw')
        IMU data mode. This is configured in Trigno Control Utility.
    host : str, optional (default: 'localhost')
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional (default: 50040)
        Port of TCU command messages.
    data_port : int, optional (default: 50042)
        Port of TCU EMG data access.
    timeout : float, optional (default: 5)
        Number of seconds before socket returns a timeout exception.
    """
    # _signals_per_channel = 9

    def __init__(
            self,
            channels,
            samples_per_read,
            zero_based=False,
            imu_mode='raw',
            host='localhost',
            cmd_port=50040,
            data_port=50044,
            timeout=5):

        self._signals_per_channel = self.get_num_spc(imu_mode)

        super(TrignoIMU, self).__init__(
            samples_per_read=samples_per_read,
            zero_based=zero_based,
            host=host,
            cmd_port=cmd_port,
            data_port=data_port,
            timeout=timeout)

        self.imu_mode = imu_mode
        self._relevant_signals_per_channel = self.get_num_relevant_spc(
            imu_mode)

        self.set_channels(channels=channels, zero_based=zero_based)

    def get_num_relevant_spc(self, imu_mode):
        if imu_mode == 'raw':
            relevant_signals_per_channel = 9
        elif imu_mode == 'quaternion':
            relevant_signals_per_channel = 4
        elif imu_mode == 'euler':
            relevant_signals_per_channel = 3
        else:
            raise ValueError("Invalid IMU mode {}.".format(imu_mode))

        return relevant_signals_per_channel

    def get_num_spc(self, imu_mode):
        if imu_mode == 'raw':
            signals_per_channel = 9
        elif imu_mode == 'quaternion':
            signals_per_channel = 5
        elif imu_mode == 'euler':
            signals_per_channel = 5
        else:
            raise ValueError("Invalid IMU mode {}.".format(imu_mode))

        return signals_per_channel


class QuattroIMU(_BaseTrignoAuxDaq, _BaseQuattro):
    """
    Delsys Quattro IMU data acquisition.

    Requires the Trigno Control Utility to be running.

    Parameters
    ----------
    channels : list or tuple
        Sensor channels to use. Each sensor has three or six IMU channels,
        depending on the selected `mode`. Only include the paired slot numbers.
        For example, if using a single sensor paired at slot 1 set
        `channels=[1]`.
    samples_per_read : int
        Number of samples per channel to read in each read operation.
    zero_based : boolean, optional (default: False)
        Whether channel numbering follows zero-based convention.
    mode : int, optional (default: 313)
        Operation sensor mode.
    host : str, optional (default: 'localhost')
        IP address the TCU server is running on. By default, the device is
        assumed to be attached to the local machine.
    cmd_port : int, optional (default: 50040)
        Port of TCU command messages.
    data_port : int, optional (default: 50043)
        Port of TCU EMG data access.
    timeout : float, optional (default: 5)
        Number of seconds before socket returns a timeout exception.
    """
    _signals_per_channel = 9

    def __init__(
            self,
            sensors,
            samples_per_read,
            zero_based=False,
            mode=313,
            host='localhost',
            cmd_port=50040,
            data_port=50044,
            timeout=5):
        super(QuattroIMU, self).__init__(
            samples_per_read=samples_per_read,
            zero_based=zero_based,
            host=host,
            cmd_port=cmd_port,
            data_port=data_port,
            timeout=timeout)

        self.sensors = sensors
        self.mode = mode
        self._relevant_signals_per_channel = self.get_num_relevant_spc(mode)

        channels = self._channels_from_sensors(
            sensors=sensors,
            zero_based=zero_based)

        self.set_channels(channels=channels, zero_based=zero_based)
        self._set_sensor_modes(sensors=sensors, mode=mode)

    def get_num_relevant_spc(self, mode):
        # https://www.delsys.com/downloads/USERSGUIDE/trigno/sdk.pdf
        raw_modes = list(range(245, 261)) + list(range(263, 295)) + \
            list(range(297, 313)) + list(range(314, 362))
        orientation_modes = [261, 295, 313]
        emg_modes = [262, 296]
        if mode in raw_modes:
            relevant_signals_per_channel = 6
        elif mode in orientation_modes:
            relevant_signals_per_channel = 4
        elif mode in emg_modes:
            relevant_signals_per_channel = 0

        return relevant_signals_per_channel

    def _channels_from_sensors(self, sensors, zero_based):
        """Maps sensor (i.e. slot) numbers to channel numbers."""
        if zero_based:
            sensors = [sensor + 1 for sensor in sensors]

        channels = []
        for sensor in sensors:
            start = self._send_cmd("SENSOR " + str(sensor) + " STARTINDEX?")
            channels.append(int(start))

        if zero_based:
            channels = [channel - 1 for channel in channels]

        return channels

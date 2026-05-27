---
standard: MIL-STD-883
method: "1021.2"
category: radiation
language: en
---

# MIL-STD-883 Method 1021.2 — DOSE RATE UPSET TESTING OF DIGITAL MICROCIRCUITS

METHOD 1021.2
DOSE RATE UPSET TESTING OF DIGITAL MICROCIRCUITS
1. PURPOSE. This test procedure defines the requirements for testing the response of packaged digital integrated
circuits to pulsed ionizing radiation. A flash x-ray or linear accelerator is used as a source of pulses of ionizing radiation.
The response may include transient output signals, changes in the state of internal storage elements, and transient current
surges at inputs, outputs, and power supply connections. The dose rate at which logic or change-of-state errors first occur
is of particular interest in many applications.
1.1 Definitions. Definitions of terms used in this procedure are given below:
a. Dose rate threshold for upset. The dose rate which causes either:
(1) A transient output upset for which the change in output voltage of an operating digital integrated circuit goes
either above or below (as appropriate) specified logic levels (see 3.2 on transient voltage criteria), and the
circuit spontaneously recovers to its preirradiation condition after the radiation pulse subsides, or
(2) A stored data or logic state upset for which there is a change in the state of one or more internal memory or
logic elements that does not recover spontaneously after the radiation pulse. However, the circuit can be
restored to its preirradiation condition by applying the same sequence of logic signals to its inputs that were
previously used to establish the preirradiation condition, or
(3) A dynamic upset which results in a change in the expected output or stored test pattern of a device that is
functionally operating during the time it is irradiated. The upset response may depend on the precise time
relationship between the radiation pulse and the operating cycle of the device. For operations requiring
many clock signals, it may be necessary to use a wide radiation pulse.
b. Dose rate. Energy absorbed per unit time per unit mass by a given material from the radiation field to which it is
exposed.
c. Combinational logic circuit. A digital logic circuit with the property that its output state is solely determined by the
logic signals at its inputs. Combinational logic circuits contain no internal storage elements. Examples of
combinational circuits include gates, multiplexers, and decoders.
d. Sequential logic circuit. A digital logic circuit with the property that its output state at a given time depends on the
sequence and time relationship of logic signals that were previously applied to its inputs. Sequential logic circuits
contain internal storage elements. Examples of sequential logic circuits include memories, shift registers,
counters, and flip-flops.
e. State vector. A state vector completely specifies the logic condition of all elements within a logic circuit. For
combinational circuits the state vector includes the logic signals that are applied to all inputs; for sequential
circuits the state vector must also include the sequence and time relationship of all input signals (this may include
many clock cycles).
METHOD 1021.2
15 November 1991
1

1.2 Interferences. There are several interferences that need to be considered when this test procedure is applied. These
include:
a. Total dose damage. Devices may be permanently damaged by total dose. This limits the number of radiation
pulses that can be applied during transient upset testing. The total dose sensitivity depends on fabrication
techniques and device technology. MOS devices are especially sensitive to total dose damage. Newer bipolar
devices with oxide-isolated sidewalls may also be affected by low levels of total dose. The maximum total dose to
which devices are exposed must not exceed 20 percent of the typical total dose failure level of the specific part
type.
b. Steps between successive radiation levels. The size of the steps between successive radiation levels limits the
accuracy with which the dose rate upset threshold is determined. Cost considerations and total dose damage
limit the number of radiation levels that can be used to test a particular device.
c. Latchup. Some types of integrated circuits may be driven into a latchup condition by transient radiation. If latchup
occurs, the device will not function properly until power is temporarily removed and reapplied. Permanent damage
may also occur, primarily due to the large amount of localized heating that results. Although latchup is an
important transient response mechanism, this procedure does not apply to devices in which latchup occurs.
Functional testing after irradiation is required to detect internal changes of state, and this will also detect latchup.
However, if latchup occurs it will usually not be possible to restore normal operation without first interrupting the
power supply.
d. Limited number of state vectors. Cost, testing time, and total dose damage usually make it necessary to restrict
upset testing to a small number of state vectors. These state vectors must include the most sensitive conditions
in order to avoid misleading results. An analysis is required to select the state vectors used for radiation testing to
make sure that circuit and geometrical factors that affect the upset response are taken into account (see 3.1).
2. APPARATUS. Before testing can be done, the state vectors must be selected for radiation testing. This requires a
logic diagram of the test device. The apparatus used for testing shall consist of the radiation source, dosimetry equipment,
a test circuit board, line drivers, cables, and electrical test instrumentation to measure the transient response, provide bias,
and perform functional tests. Adequate precautions shall be observed to obtain an electrical measurement system with
ample shielding, satisfactory grounding, and low noise from electrical interference or from the radiation environment.
2.1 Radiation source. The radiation source used in this test shall be either a flash x-ray machine (FXR) used in the
photon mode or a linear accelerator (LINAC) used in the electron beam mode. The LINAC beam energy shall be greater
than 10 MeV. The radiation source shall provide a uniform (within 20 percent) radiation level across the area where the
device and the dosimeter will be placed. The radiation pulse width for narrow pulse measurements shall be between 10 and
50 ns. For narrow pulse measurements either a LINAC or FXR may be used. Wide pulse measurements (typically 1 - 10
µs) shall be performed with a LINAC. The pulse width for LINAC irradiations shall be specified. The dose rate at the
location of the device under test shall be adjustable between 106 and 1012 rads(Si)/s (or as required) for narrow pulse
measurements and between 105 and 1011 rads(Si)/s (or as required) for wide pulse measurements. Unless otherwise
specified, a test device exposed to a total dose that exceeds 20 percent of the total dose failure level shall be considered as
destructively tested and shall be removed from the lot (see 1.2a).
2.2 Dosimetry equipment. Dosimetry equipment must include a system for measuring total dose, such as a
thermoluminescent dosimeter (TLD) or calorimeter, a pulse shape monitor, and an active dosimeter that allows the dose rate
to be determined from electronic measurements, e.g., a p-i-n detector, Faraday cup, secondary emission monitor, or current
transformer.
METHOD 1021.2
15 November 1991
2

2.3 Test circuit. The test circuit shall contain the device under test, wiring, and auxiliary components as required. It shall
allow for the application of power and bias voltages or pulses at the device inputs to establish the state vector. Power
supply stiffening capacitors shall be included which keep the power supply voltage from changing more than 10 percent of
its specified value during and after the radiation pulse. They should be placed as close to the device under test as possible,
but should not be exposed to the direct radiation beam. Provision shall be made for monitoring specified outputs.
Capacitive loading of the test circuit must be sufficiently low to avoid interference with the measurement of short-duration
transient signals. Generally a line driver is required at device outputs to reduce capacitive loading. Line drivers must have
sufficient risetime, linearity, and dynamic range to drive terminated cables with the full output logic level. The test circuit
shall not affect the measured output response over the range of expected dose rates and shall not exhibit permanent
changes in electrical characteristics at the expected accumulated doses. It must be shielded from the radiation to a
sufficient level to meet these criteria.
Test circuit materials and components shall not cause attenuation or scattering which will perturb the uniformity of the beam
at the test device position (see 2.1 for uniformity). The device under test shall be oriented so that its surface is
perpendicular to the radiation beam.
2.4 Cabling. Cabling shall be provided to connect the test circuit board, located in the radiation field, to the test
instrumentation located in the instrumentation area. Coaxial cables, terminated in their characteristic impedance, shall be
used for all input and output signals. Double shielded cables, triax, zipper tubing or other additional shielding may be
required to reduce noise to acceptable levels.
2.5 Transient signal measurement. Oscilloscopes or transient digitizers are required to measure transient output
voltages, the power supply current and the dosimeter outputs. The risetime of the measuring instrumentation shall be less
than 10 ns for pulse widths greater than 33 ns or less than 30 percent of the radiation pulse width for pulse widths less than
33 ns.
2.6 Functional testing. Equipment is also required for functional testing of devices immediately after the radiation pulse in
the radiation test fixture. This equipment must contain sources to drive inputs with specified patterns, and comparison
circuitry to determine that the correct output patterns result. This equipment may consist of logic analyzers, custom circuitry,
or commercial integrated circuit test systems. However, it must be capable of functioning through long cables, and must
also be compatible with the line drivers used at the outputs of the device in the test circuit.
2.7 General purpose test equipment. Power supplies, voltmeters, pulse generators, and other basic test equipment that
is required for testing is general purpose test equipment. This equipment must be capable of meeting the test requirements
and should be periodically calibrated in accordance with ANSI/NCSL Z540-1 or equivalent.
3. PROCEDURE. An outline of the procedure is as follows: a) determine the state vectors (or sequence of test vectors
for a dynamic test) in which the device will be irradiated; b) following the test plan, set up the test fixture, functional test
equipment, and transient measurement equipment; c) set up and calibrate the radiation source; d) perform a noise check on
the instrumentation; and e) test devices at a sequence of radiation pulses, determining the transient response at specified
dose rates. The dose rate upset level can be determined by measuring the transient response at several dose rates, using
successive approximation to determine the radiation level for dose rate upset.
3.1 State vector selection. Two approaches can be used to select the state vectors in which a device is to be irradiated:
a. Multiple output logic states. Partition the circuit into functional blocks. Determine the logic path for each output,
and identify similar internal functions. For example, a 4-bit counter can be separated into control, internal flip-flop,
and output logic cells. Four identical logic paths exist, corresponding to each of the four bits. Determine the total
number of unique output logic state combinations, and test the circuit in each of these states. For the counter
example this results in 16 combinations so that the upset must be determined for each of these 16 state vectors.
METHOD 1021.2
15 November 1991
3

b. Topological analysis approach. If a photomicrograph of the circuit is available, the number of required states can
be reduced by examining the topology of the internal circuits. This allows one to eliminate the need to test paths
with the same output state which have identical internal geometries. For the counter, this reduces the required
number of states to two. This approach is recommended for more complex circuits where the multiple output logic
approach results in too many required state vectors.
3.2 Transient output upset criteria. The transients that are permitted at logic outputs depend on the way that the system
application allocates the noise margin of digital devices. Most systems use worst-case design criteria which are not directly
applicable to sample testing because the samples represent typical, not worst-case parts, and have higher noise margins.
For example, although the logic swing of TTL logic devices is typically greater than 2 volts, the worst-case noise margin is
specified at 400 mV. In a typical system, much of this noise margin will be required for aberrations and electrical noise,
leaving only part of it, 100 mV to 200 mV, for radiation-induced transients. Thus, the allowable voltage transient is far lower
than the typical logic signal range. Loading conditions also have a large effect on output transients.
However, transient upset testing is usually done at a fixed temperature under conditions that are more typical than they are
worst-case. Thus, the noise margin during testing is much greater. The recommended default condition if not specified by
the system is a transient voltage exceeding 1 V for CMOS or TTL logic devices with 5 V (nominal) power supply voltage, and
30 percent of the room-temperature logic level swing for other technologies such as ECL, open collector devices, or
applications with other power supply voltages. Default loading conditions are minimum supply voltage and maximum fanout
(maximum loading).
The time duration of transient upset signals is also important. If the duration of the transient voltage change is less than the
minimum value required for other circuits to respond to it, the transient signal shall not be considered an upset. The
minimum time duration shall be one-half the minimum propagation delay time of basic gate circuits from the circuit
technology that is being tested.
Testing criteria may also be established for other parameters, such as the power supply current surge. Output current is
also important for tri-state or uncommitted ("open-collector") circuits. These criteria must be specified by the test plan, and
are normally based on particular system requirements.
3.3 Test plan. The test plan must include the following:
a. Criteria for transient voltage upset, output current, and power supply current, as applicable.
b. Power supply and operating frequency requirements.
c. Loading conditions at the outputs.
d. Input voltage conditions and source impedance.
e. Functional test approach, including dynamic upset, if applicable.
f. Radiation pulse width(s).
g. Sequence used to adjust the dose rate in order to determine the upset threshold by successive approximation.
h. State vectors used for testing (determined from 3.1).
i. Radiation levels to be used for transient response measurements, if applicable.
j. A recommended radiation level at which to begin the test sequence for transient upset measurements, if
applicable.
k. The temperature of the devices during testing (usually 25°C ±5°C).
METHOD 1021.2
15 November 1991
4

3.4 Test circuit preparation. The test circuit shall be assembled including a test circuit board, line drivers, electrical
instruments, functional test equipment, transient measurement equipment, and cables to provide the required input biasing,
output monitoring, and loading.
3.5 Facility preparation. The radiation source shall be adjusted to operate in the specified mode and provide a radiation
pulse width within the specified width range. The required dosimeters shall be installed as close as practical to the device
under test. If special equipment is needed to control the temperature to the value specified in the test plan, this equipment
must be assembled and adjusted to meet this requirement.
3.6 Safety requirements. The health and safety requirements established by the local Radiation Safety Officer or Health
Physicist shall be observed.
3.7 Test circuit noise check. With all circuitry connected, a noise check shall be made. This may be done by inserting a
resistor circuit in place of the test device. Resistor values chosen shall approximate the active resistance of the device
under test. A typical radiation pulse shall be applied while the specified outputs are monitored. If any of the measured
transient voltages are greater than 10 percent of the expected parameter response, the test circuit is unacceptable and shall
not be used without modification to reduce noise.
3.8 Bias and load conditions. Unless otherwise specified, the power supply shall be at the minimum allowed value. Input
bias levels shall be at worst-case logic levels. Outputs shall be loaded with the maximum load conditions in both logic states
(usually equivalent to maximum circuit fanout).
3.9 Temperature. The temperature of the devices during test should be measured with an accuracy of ±5°C unless
higher accuracy is required in the test plan.
3.10 Procedure for dose rate upset testing. The device to be tested shall be placed in the test socket. The required
pulse sequence shall be applied so that the device is in the state specified by the first of the state vectors in 3.1.
a. Set the intensity of the radiation source to the first radiation test level specified in the test plan. Expose the device
to a pulse of radiation, and measure the transient output responses and power supply current transient. For
sequential logic circuits, perform a dynamic functional test to see if changes occurred in internal logic states.
b. Repeat 3.10a for all other state vectors and radiation levels specified in the test plan.
3.11 Radiation exposure and test sequence for upset threshold testing. The device to be tested shall be placed in the
test socket. The required pulse sequence shall be applied so that the device is in the state specified by the first of the state
vectors determined in 3.1, or is operating with the specified test vector sequence for dynamic upset.
a. Set the intensity of the radiation source to the initial level recommended in the test plan, and expose the device to
a pulse of radiation. Determine whether a stored data upset, logic state upset, or dynamic upset occurs, as
appropriate.
b. If no upset occurred, increase the radiation level according to the sequence specified in the test plan; if an upset
is observed decrease the radiation level. After the radiation source is adjusted to the new intensity, reinitialize the
part to the required state vector, expose it to an additional pulse, and determine whether or not upset occurred.
Continue this sequence until the upset response threshold level is bracketed with the resolution required in the
test plan.
c. The power supply peak transient current shall be monitored and recorded during radiation testing unless it is not
required by the test plan.
d. Repeat test sequences 3.11a through 3.11c for all of the state vectors.
METHOD 1021.2
15 November 1991
5

3.12 Report. As a minimum the report shall include the following:
a. Device identification.
b. Test date and test operator.
c. Test facility, radiation source specifications, and radiation pulse width.
d. Bias conditions, output loading, and test circuit.
e. Description of the way in which state vectors for testing were selected.
f. State vectors used for radiation testing and functional test conditions for each state vector.
g. Criteria for transient output upset.
h. Records of the upset threshold and power supply current for each state vector.
i. Equipment list.
j. Results of the noise test.
k. Temperature (see 3.9).
4. SUMMARY. The following details shall be specified.
a. Device type and quantity to be tested.
b. Test circuit to be used, including output loading impedance.
c. State vectors to be used in testing and device output pins to be monitored.
d. Functional test sequence.
e. Power supply voltage and bias conditions for all pins.
f. Pulse width of the radiation source (see 2.1).
g. The method of selecting steps between successive irradiation levels and the required resolution.
h. Restrictions on ionizing (total) dose if other than that specified in 3.1.
i. Temperature of the devices during testing.
j. Requirement for measuring and recording power supply peak transient current (see 3.11c).
k. Failure criteria for transient output voltage upset.
l. Failure criteria for power supply current and output current, if applicable.
METHOD 1021.2
15 November 1991
6

---
standard: MIL-STD-883
method: "1023.2"
category: radiation
language: en
---

# MIL-STD-883 Method 1023.2 — Dose Rate Response and Threshold for Upset

METHOD 1023.2
Dose Rate Response and Threshold for Upset
of Linear Microcircuits
1. PURPOSE. This test procedure defines the requirements for measuring the dose rate response and upset threshold
of packaged devices containing analog functions when exposed to radiation from a flash X-ray source or from a linear
accelerator. This procedure addresses the measurement of dose rate response characteristics of a linear circuit, excluding
latchup which is addressed in MIL-STD-883 Test Method 1020.
1.1 Definitions. The following are the definitions of terms used in this method:
a. Dose rate response. The transient changes which occur in the operating parameters or in the output signal of an
operating linear microcircuit when exposed to a pulse of ionizing radiation.
b. Dose rate. Energy absorbed per unit time and per unit mass by a given material from the radiation field to which it
is exposed. Units are specified in Gray (Gy) per second (s) in the material of interest, e.g., Gy(Si)/s, Gy(SiO )/s,
2
Gy(GaAs)/s, etc.
c. Dose rate induced upset. An upset has occurred when the radiation induced transient change in a specified
parameter (e.g., in output voltage, supply current, output signal waveform) exceeds a predetermined level.
d. Upset threshold. The upset threshold is the minimum dose rate at which the device upsets. However, the
reported measured upset threshold shall be the maximum dose rate at which the device does not upset and which
the transient disturbance of the output waveform and/or supply current remains within the specified limits.
1.2 Test plan. Prior to dose rate testing, a test plan shall be prepared which describes the radiation source, the
dosimetry techniques, test equipment, the device to be tested, test conditions, and any unique testing considerations. A
detailed procedure for each device type to be tested shall be prepared, either as part of the test plan, or in separate test
procedure documents. The procedure shall include bias conditions, test sequence, schematics of the test setup and specific
functions to be tested. The test plan shall be approved by the acquiring activity, and as a minimum, the items listed below
shall be provided in the test plan or procedure:
a. Device types, including package types, manufacturer, date codes, and quantities to be tested.
b. Traceability requirements, such as requirements for serialization, wafer or lot traceability, etc.
c. Requirements for data reporting and submission.
d. Block diagram or schematic representation of test set up.
e. List of equipment used in the testing and calibration compliance requirements as required.
f. Test conditions, e.g., bias voltage, temperature, etc.
g. Electrical parameters to be monitored and device operating conditions, including functional test requirements
before, during and after the radiation pulse. Test patterns to be used for devices with storage elements, or
devices with input pattern sensitivity shall also be specified.
h. Group A electrical test requirements for pre- and post-dose rate testing, when applicable, to include test limits and
failure criteria.
i. Radiation test parameters such as pulse width(s), radiation dose(s) per pulse and dose rate range(s).
j. Total ionizing dose limit acceptable for each device type.
k. Upset and failure criteria, e.g., effective number of bits (ENOB) or missing codes in analog to digital converters
(ADCs), delta VOH or Vref, time to recovery, output waveform distortion in shape or frequency, etc.
METHOD 1023.2
19 August 1994
1

1.3 Formulation of the upset criteria. The upset criteria are usually generated from characterization data at the dose rate
of interest. Upset criteria can sometimes be determined by analysis/simulation (SPICE or equivalent computer code) of the
application circuit, if the code has been verified to agree with experimental data for similar circuits and exposure conditions.
1.4 Specification of the upset criteria. Once formulated, the upset criteria shall be specified in the detailed specification.
The upset criteria may consist of the following (a waveform may be included denoting the acceptable boundaries):
a. Measurement circuit to which criteria apply.
b. Peak amplitude of tolerable transient change in output voltage.
c. Allowable duration of transient output change (recovery time).
d. Limiting value for the surge in power supply current and recovery characteristics.
e. Steady state (return to normalcy) level of the output voltage following recovery.
f. ENOB or missing codes for ADCs.
g. Delta parameters such as Vref or VOH.
h. Device saturation time.
2. APPARATUS. The apparatus shall consist of the radiation source, dosimetry equipment, remote test circuit to include
signal recording devices, cabling, line drivers, interconnect fixture, and exposure board. Adequate precautions shall be
observed to obtain an electrical measurement system with sufficient insulation, ample shielding, satisfactory grounding and
low noise from electrical interference or from the radiation environment (see section 3.7.3).
2.1 Radiation Source. Either of two radiation sources shall be used for dose rate testing: 1) a flash x-ray machine
(FXR), or 2) an electron linear accelerator (LINAC). The FXR shall be used in the x-ray mode and the LINAC in the electron
(e-beam) mode. Unless otherwise specified, the FXR peak charging voltage shall be 2 MV or greater, and the LINAC beam
energy shall be 10 MeV or greater. The uniformity of the radiation field in the device irradiation volume shall be + 15% as
measured by the dosimetry system. The dose per radiation exposure shall be as specified in the test plan or procedure.
2.2 Dosimetry System. A dosimetry system shall be used which provides a measurement accuracy within + 15 percent.
A calibrated PIN diode may be used to obtain both the shape of the radiation pulse and the dose. The following American
Society for Testing and Materials (ASTM) standards or their equivalent may be used:
ASTM E 526 Standard Method for Measuring Dose for Use in Linear Accelerator Pulsed Radiation Effects Tests.
ASTM E 666 Standard Method for Calculation of Absorbed Dose from Gamma or X Radiation.
ASTM E 668 Standard Practice for the Application of Thermo-luminescence Dosimetry (TLD) Systems for Determining
Absorbed Dose in Radiation Hardness Testing of Electronic Devices.
These methods describe techniques to determine the absorbed dose in the material of interest. Device packaging material
and thickness should be considered in determining the dose to the DUT. For FXR tests, dose enhancement effects of the
package shall be considered. Dosimetry techniques shall be reported in the test report as well as device packaging
material, thickness and dose enhancement effects, if applicable.
METHOD 1023.2
19 August 1994
2

2.3 Dose Rate Test System. The instrumentation shall be capable of establishing the required test conditions and
measuring and recording the required parameters in the specified time frame. Components other than the device under test
(DUT) shall be insensitive to the expected radiation levels, or they shall be shielded from the radiation. The system used for
dose rate testing shall contain the following elements:
2.3.1 Remote Test circuit. The remote portion of the test circuit includes power sources, input and control signal
generators, instrumentation for detecting, measuring and recording transient and steady state response, and may also
include automated test equipment (ATE). The remote portion of the test equipment is shielded from radiation and from
radiation induced electromagnetic fields. Specified signals shall be measured and recorded during the radiation pulse, and
the logic pattern shall be verified after the pulse (when applicable).
2.3.2 Interconnect fixture. The interconnecting fixture is located in the radiation exposure chamber and is connected to
the remote portion of the test circuit via the cabling system. It serves as a power and signal distribution box and contains
the line drivers that buffer the various DUT output signals. The characteristics of the line drivers (e.g., linearity, dynamic
range, input capacitance, transient response and radiation response) shall be such that they accurately represent the
response of the DUT output. The interconnect fixture shall be located as close as practical to the exposure fixture, and must
be appropriately shielded against scattered radiation fields so that radiation induced effects do not adversely affect the
fidelity of the output response being measured.
2.3.3 Test circuit. The test circuit for each device type shall provide worst case bias and load conditions for the DUT, and
shall enable in-situ functional testing of the DUT as specified in the test plan or procedure. The test circuit accommodates
the DUT, output loads, and the supply stiffening capacitors connected directly to the DUT supply pins or its socket (see
2.3.4). To avoid ground loops, there shall be only one ground plane (or ground rings connected to a single ground) on the
test circuit. Test Circuit parasitic resistance shall be kept to a minimum.
2.3.4 Stiffening capacitors. A high frequency capacitor shall be placed at each bias supply pin of the DUT with lead
lengths as short as practicable. These capacitors should be large enough such that the power supply voltage drop at the
DUT is less than 10% during the radiation pulse (typical values are between 4.7 and 10 µF). In parallel with this capacitor
should be a low inductance capacitor (e.g., 0.1 µF), again as close as possible to the supply pin and with lead lengths as
short as practical. In addition, for each supply line into the DUT, a larger capacitor, > 100 µF, may be placed a short
distance away from the DUT and shielded from radiation.
2.3.5 Current Limiting Series resistor. A current limiting resistor in series with the power supply may only be used with
prior approval of the acquiring activity. Note that a current limiting resistor may degrade the upset performance of the DUT.
2.3.6 Timing control. A timing control system shall be incorporated into the test system such that post-irradiation in-situ
functional testing is performed at the specified time, and that recovery of the signal and supply current can be monitored.
2.4 Cabling. The remote test circuit shall be connected to the interconnect and exposure fixtures by means of shielded
cables terminated in their characteristic impedance. Additional shielding provisions (e.g., doubly shielded cables, triax,
zipper tubing, aluminum foil) may be required to reduce noise to acceptable levels.
2.5 Measuring and recording equipment. Oscilloscopes or transient waveform digitizers shall be used to measure and
record the transient signal and the recovery period of the output voltage and supply current. The rise time of these
instruments shall be such that they are capable of accurately responding to the expected pulse width(s).
3. PROCEDURE.
3.1 Device identification. In all cases, devices shall be serialized, and the applicable recorded test data shall be
traceable to each individual device.
METHOD 1023.2
19 August 1994
3

3.2 Radiation safety. All personnel shall adhere to the health and safety requirements established by the local radiation
safety officer or health physicist.
3.3 Stress limits.
3.3.1 Total ionizing dose limit. Unless otherwise specified, any device exposed to more than 10% of its total ionizing
dose limit shall be considered to have been destructively tested. The total dose limit shall be determined (or data obtained)
for each device type to be tested. The total ionizing dose limit shall be specified in the test plan.
3.3.2 Burnout Limit. A device exposed to greater than 10% of the level at which photocurrent burnout occurs shall be
considered destructively tested. The burnout level shall be specified in the test plan/procedure. The burnout level may be
specified as the maximum dose rate level at which the device type has been tested and does not burnout. Note that dose
rate testing causes surge currents ranging from 20 ns to 500 ns (typically) in duration, which may exceed the manufacturers'
maximum ratings for current and power for that time period.
3.4 Characterization testing. Characterization tests shall be performed or data obtained to determine device performance
as a function of dose rate and to establish requirements for production testing, if applicable. The following are examples of
information gained from characterization testing:
a. Parameter behavior over dose rate and pulse width.
b. Upset threshold as a function of radiation dose rate and pulse width.
c. Determination of susceptible circuit conditions.
d. Identification of the most susceptible circuits of a device, and the appropriate outputs to monitor.
e. Effect of temperature on upset or failure.
f. Upset, recovery time and failure criteria to be specified in the device specification or drawing.
g. Group A electrical parameter degradation subsequent to dose rate testing.
h. Worst case power supply voltage.
i. Maximum surge currents and duration, and photocurrent burnout level.
3.5 Production testing. Prior to production testing, characterization testing shall be performed or characterization data
obtained for each device type. The results of the characterization tests (paragraph 3.4), or the existing data, will be used to
develop the requirements for the production tests. These requirements are specified in the applicable test plan or procedure
and include those items listed in paragraph 1.2.
3.5.1 General requirements for production tests. Production tests shall be performed at the specified dose rates (and
pulse widths), with bias and load conditions as specified in the test plan or procedure. The measured response shall be
compared to the upset criteria and determination of pass/fail shall be made. Devices having storage elements shall be
loaded with the applicable test pattern prior to exposure and post-exposure functional test shall be performed to the extent
necessary to verify the stored pattern.
3.6 Testing of Complex Linear Devices. Testing of complex linear devices, such as analog to digital and digital to
analog converters, shall be performed using the necessary (as specified in the test plan or procedure) exposure conditions
to ensure adequate coverage. Often, four or more exposure conditions are required. To the greatest extent practical, the
most susceptible exposure conditions (i.e., most favorable for upset to occur) shall be used. For linear devices that have
storage elements, each exposure state shall consist of a stored test pattern plus the external bias. Each test pattern shall
be loaded prior to exposure, and following the application of the radiation pulse, functional testing of the device must be
performed to the extent necessary to verify the pattern.
METHOD 1023.2
19 August 1994
4

3.7 Dose Rate Test Sequence.
3.7.1 Facility Preparation. The radiation source shall be adjusted to operate in the specified mode and provide a radiation
pulse within the specified pulse width range. The required dosimeters shall be installed as close as practical to the DUT.
3.7.2 Test Circuit preparation. The dose rate test system, including all test circuitry, cables, monitoring and recording
equipment shall be assembled to provide the specified bias and load conditions and output monitoring. The test circuit shall
be placed in position such that the DUT will receive the specified dose. Unless otherwise specified, dose rate testing shall
be performed at 25° + 5°C. (The test temperature shall be specified in the test plan/procedure.)
3.7.3 Test circuit noise check. With all circuitry connected, a noise check, including radiation induced noise, shall be
made. Noise signals shall be kept as low as practicable. The circuitry and cabling system shall be modified until the noise
signals are below an acceptable level (usually less than 10% of the expected response).
3.7.4 Test Procedure.
CAUTION: Exercise caution when handling devices, particularly with regard to pin alignment in the and holding fixture
and when installing devices in the test circuit. Ensure that voltages are off before inserting the DUT. Observe ESD
handling procedures for the class of devices being tested, as appropriate.
Step 1: Adjust timing control system to provide the required time interval between radiation pulse and
post-irradiation measurements.
Step 2: Remove bias voltages and install a control sample device (same type as devices to be tested).
Step 3: Turn on bias voltages and verify proper device function in accordance with performance
requirements.
Step 4. Verify proper operation of all recording, monitoring and timing control equipment. Monitor and
record noise level and temperature.
Step 5. Remove bias voltages and control device, in that order.
Adjust the radiation source to operate in the specified mode to deliver the specified dose. Verify as follows:
Step 6: Put dosimetry in position and expose to radiation pulse. Verify that the dose recording
equipment is working properly and that the appropriate dose was delivered.
When the dose rate test system, radiation source and dosimetry system have been verified to be working properly, continue
as follows for each device type to be tested:
Step 7: Ensure bias is removed from the test circuit and install DUT.
Step 8: Bias the device and load the test patterns (if applicable) in accordance with the test plan or
procedure. Verify proper device functional operation.
Step 9: Expose the DUT to the radiation pulse and measure the response of the specified outputs, as
well as the recovery characteristics.
Step 10: Compare the DUT response to upset criteria, if applicable.
Repeat steps 8-10 for each exposure state and for each radiation dose rate.
Step 11: Remove bias and DUT in that order.
Note that the upset threshold shall be reported as the maximum dose rate at which the DUT does not upset.
METHOD 1023.2
19 August 1994
5

4. Test Report. A dose rate test report shall be prepared and shall include the following (as a minimum):
a. Device identification, including manufacturer, wafer lot and/or inspection lot traceability information, pre-radiation
history (e.g., class level S, class level B, prototype, etc).
b. Radiation test facility, type of source, pulse width, dosimetry data including pulse waveform.
c. Test date, test operator's name and organization.
d. Results of the noise test.
e. Device response data, listed by device serial number, including output and supply recovery waveforms, and dose
per pulse for each device.
f. Power supply droop during pulse.
g. Post-exposure functional test data if applicable.
h. All information included in the test plan/procedure (may be referenced or appended to test report), and any
deviations from the approved test plan/procedure.
i. Package material and thickness, and effect of package material on dose to the device (see paragraph 2.2).
5. SUMMARY. The applicable device specification or drawing shall specify the following (as applicable):
a. Device types and quantities to be tested.
b. Traceability (device number, wafer/lot number, etc.) requirements and requirements for data reporting and
submission.
c. Electrical configuration of the DUT during exposure (include schematic of exposure configuration).
d. Sequence of exposure conditions and logical test patterns.
e. Outputs to be monitored and recorded.
f. Dose rate level(s) and pulse width(s).
g. Criteria for upset and recovery, steady state value of recovered outputs and/or supply current. Include sample
waveforms if necessary.
h. Upset threshold and failure level (if applicable).
i. Post exposure functional test necessary to verify the stored pattern, and maximum time interval between
application of the radiation pulse and start of functional test.
j. Total ionizing dose limit and burnout level for each device type.
k. Maximum current limiting resistance in series with the power supply in the application (if applicable), and
allowable resistance in the test circuit (paragraph 2.3.5).
l. Requirements for Group A electrical testing pre- and post-radiation testing, if applicable.
m. Test instrument requirements, if other than those indicated above.
n. Requirements for characterization, recharacterization and analysis.
METHOD 1023.2
19 August 1994
6

APPENDIX A
A1. This appendix provides an example of the specification of test details for an operational amplifier. Because the test
conditions depend both on the type of device and on the specific application, this example shall not be considered as
suitable for use in any given case. It is provided only as an illustration of the use of this test method.
A2. Test specification, method 1023:
a. Type 741 operational amplifier, in 8-pin TO-5 package.
b. Test circuit as given on figure 1023-1. Leave pins 1, 5, and 8 unconnected.
c. V+=9.0 + 0.2 V; V-=-9.0 + 0.2 V; input signal 280 mV +5% peak to peak, 2000 +50 Hz.
d. Monitor pin 6 and the power supply current.
e. Standard noise limits apply.
f. Pulse width: 20 ns (Full width half maximum).
g. Total Ionizing dose shall not exceed 10 Gy(Si).
h. Test at a dose rate of 105 +30% Gy(Si)/s.
i. Test temperature shall be ambient (25° +5°C).
j. Pass/Fail Criteria: Power supply currents and the output signal shall return to within 10% of the
pre-rad levels within 1 ms of the radiation pulse.
k. This test is considered a destructive test.
METHOD 1023.2
19 August 1994
7

FIGURE 1023-1. Example of test circuit for OP-AMP.
METHOD 1023.2
19 August 1994
8

APPENDIX B
B1. This appendix provides an example of the specification of test details for an analog to digital converter (ADC).
Because the test conditions depend both on the type of device and on the specific application, this example shall not be
considered as suitable for use in any given case. It is provided only as an illustration of the use of this test method.
B2. Test specification, method 1023:
a. Type ADC (n=# bits=12), 40 pin ceramic DIP.
b. The test circuit is given in Figure 1023-2, and an overview of the test setup is provided in Figure 1023-3.
The storage RAM must write at a speed (taa>tclk) exceeding the DUT clock frequency, and provide an interface to
the controller, and be capable of storing a trigger pulse from the radiation source. Note that if the data ready line of
the ADC is used, it must be monitored separately, as it may also upset.
c. A minimum of 3 input voltages shall be tested. Adjust input bias to center output code on:
1. Midscale (2n/2)
2. Fullscale - 10% (2n-0.1*2n)
3. Zero + 10% (0+0.1*2n)
d. As a minimum, perform tests at 10 MHz and 1 MHz (Fmax and 0.1*Fmax).
e. Upset Criteria: Determination of the upset threshold shall be determined by statistical analysis comparing the pre-
shot ADC output codes with the data taken during and immediately after the shot. The time to recover (within
20%) shall also be determined by comparing the pre-rad data with the post-rad data.
f. Pulse width: 20 ns (Full width half maximum).
g. Test at dose rates ranging from 102 - 107 Gy(Si)/s to establish upset threshold.
h. Total ionizing dose shall not exceed 500 Gy(Si).
i. Test at ambient temperature (25° +5°C).
j. After completion of the upset tests, test up to the machine maximum dose rate to determine if devices burn out.
This test is a destructive test.
METHOD 1023.2
19 August 1994
9

FIGURE 1023-2. Example of test circuit for ADC (C1 = 4.7 µF, C2 = 0.1 µF).
METHOD 1023.2
19 August 1994
10

FIGURE 1023-3. Schematic of example test setup for ADC.
METHOD 1023.2
19 August 1994
11

METHOD 1023.2
19 August 1994
12

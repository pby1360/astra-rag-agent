---
standard: MIL-STD-883
method: "1012.1"
category: temperature
language: en
---

# MIL-STD-883 Method 1012.1 — THERMAL CHARACTERISTICS

METHOD 1012.1
THERMAL CHARACTERISTICS
1. PURPOSE. The purpose of this test is to determine the thermal characteristics of microelectronic devices. This
includes junction temperature, thermal resistance, case and mounting temperature and thermal response time of the
microelectronic devices.
1.1 Definitions. The following definitions and symbols shall apply for the purpose of this test:
a. Case temperature, T , in °C. The case temperature is the temperature at a specified accessible reference point
C
on the package in which the microelectronic chip is mounted.
b. Mounting surface temperature, T , in °C. The mounting surface temperature is the temperature of a specified
M
point at the device-heat sink mounting interface (or primary heat removal surface).
c. Junction temperature, T, in °C. The term is used to denote the temperature of the semiconductor junction in the
J
microcircuit in which the major part of the heat is generated. With respect to junction temperature measurements,
T is the peak temperature of an operating junction element in which the current distribution is nonuniform,
J(Peak)
T is the average temperature of an operating junction element in which the current distribution is nonuniform,
J(Avg)
and T is the temperature in the immediate vicinity within six equivalent radii (an equivalent radius is the
J(Region)
radius of a circle having the same area as contained in a junction interface area) of an operating junction. In
general T <T <T . If the current distribution in an operating junction element is uniform then T
J(Region) J(Avg) J(Peak) J(Avg)
< T .
J(Peak)
d. Thermal resistance, junction to specified reference point, RθJR , in °C/W. The thermal resistance of the microcircuit
is the temperature difference from the junction to some reference point on the package divided by the power
dissipation P .
D
e. Power dissipation, P , in watts, is the power dissipated in a single semiconductor test junction or in the total
D
package, P .
D(Package)
f. Thermal response time, t , in seconds, is the time required to reach 90 percent of the final value of junction
JR
temperature change caused by the application of a step function in power dissipation when the device reference
point temperature is held constant. The thermal response time is specified as t , t , or t to
JR(Peak) JR(Avg) JR(Region)
conform to the particular approach used to measure the junction temperature.
g. Temperature sensitive parameter, TSP, is the temperature dependent electrical characteristic of the
junction-under test which can be calibrated with respect to temperature and subsequently used to detect the
junction temperature of interest.
2. APPARATUS. The apparatus required for these tests shall include the following as applicable to the specified test
procedures.
a. Thermocouple material shall be copper-constantan (type T) or equivalent, for the temperature range -180°C to
+370°C. The wire size shall be no larger than AWG size 30. The junction of the thermocouple shall be welded to
form a bead rather than soldered or twisted. The accuracy of the thermocouple and associated measuring system
shall be ±0.5°C.
b. Controlled temperature chamber or heat sink capable of maintaining the specified reference point temperature to
within ±0.5°C of the preset (measured) value.
METHOD 1012.1
4 November 1980
1

c. Suitable electrical equipment as required to provide controlled levels of conditioning power and to make the
specified measurements. The instrument used to electrically measure the temperature-sensitive parameter shall
be capable of resolving a voltage change of 0.5 mV. An appropriate sample-and-hold unit or a cathode ray
oscilloscope shall be used for this purpose.
d. Infrared microradiometer capable of measuring radiation in the 1 to 6 micrometer range and having the ability to
detect radiation emitted from an area having a spatial resolution of less than 40 micrometers (1.6 mils) diameter at
its half power points and a temperature resolution (detectable temperature change) of 0.5°C at 60°C.
NOTE: May be a scanning IR microradiometer.
e. A typical heat sink assembly for mounting the microelectronic device-under test is shown on figure 1012-1. The
primary heat sink is water cooled and has a thermocouple sensor for inlet and outlet water temperature as shown
in figure 1012-1a.
An adapter heat sink, as shown on figure 1012-1b is fastened to the top surface of the primary heat sink, and has a special
geometry to handle specific size packages, e.g., flat packs, dual-in-line packages (small and large size) and TO-5 cans.
This adapter provides a fairly repeatable and efficient interface between the package and the heat sink; the heat sink
temperature is determined from a thermocouple peened into the underside of the adapter-near the package.
The adapter also contains the socket or other electrical interconnection scheme. In the case of the flat pack adapter heat
sink, the package is dropped into a special slotted printed circuit board (PCB) to register the leads with runs on the PCB;
toggle clamps then provide a pressure contact between the package leads and the PCB runs. Dual-in-line and axial lead
packages plug into a regular socket.
The thermal probe assembly is shown on figure 1012-1b. In practice, the pressure adjustment cap is adjusted so the disk at
the probe tip contacts the bottom surface of the package (chip carrier) with a predetermined force. A silicone grease (about
25-50 mm thick) is used at this interface to provide a reliable thermal contact.
3. PROCEDURE.
3.1 Direct measurement of reference point temperature, T or T . For the purpose of measuring a microelectronic device
C M
thermal resistance or thermal response time, the reference point temperature shall be measured at the package location of
highest temperature which is accessible from outside the package. In general, that temperature shall be measured on the
surface of the chip carrier directly below the chip. The location selected shall be as near the chip as possible and
representative of a temperature in the major path of heat flow from the chip to the heat sink. The surface may be altered to
facilitate this measurement provided that such alteration does not affect the original heat transfer paths and, hence, the
thermal resistance, within the package by more than a few percent.
3.1.1 Case temperature, T . The microelectronic device under test shall be mounted on a temperature controlled heat
C
sink so that the case temperature can be held at the specified value. A thermocouple shall be attached as near as possible
to the center of the bottom of the device case directly under the chip or substrate. A conducting epoxy may be used for this
purpose. In general, for ambient cooled devices, the case temperature should be measured at the spot with the highest
temperature. The thermocouple leads should be electrically insulated up to the welded thermocouple bead. The
thermocouple bead should be in direct mechanical contact with the case of the microelectronic device under test.
3.1.2 Mounting surface temperature, T . The mounting surface temperature is measured directly below the primary heat
M
removal surface of the case. It is measured with a thermocouple at or near the mounting surface of the heat sink. A typical
mounting arrangement is shown on figure 1012-2. The surface of the copper mounting base shall be nickel plated and free
of oxides.
METHOD 1012.1
4 November 1980
2

The thermocouple hole shall be drilled into the mounting base such that the thermocouple lead is directly below the area on
the case of interest. It is recommended that the thermocouple be secured into the mounting base with a thermal conducting
adhesive (or solder) and that particular attention be paid to minimizing air voids around the ball of the thermocouple. A
thermal conducting compound (or adhesive) should be used at the interface of the mounting base and the device under test.
3.2 Thermal resistance, junction to specified reference point, RθJR .
3.2.1 General considerations. The thermal resistance of a semiconductor device is a measure of the ability of its carrier
or package and mounting technique to provide for heat removal from the semiconductor junction.
The thermal resistance of a microelectronic device can be calculated when the case temperature and power dissipation in
the device, and a measurement of the junction temperature are known. The junction with the greatest power dissipation
density (watts/mm2) shall be selected for measurement since that junction will generally have the highest temperature on the
chip. If the leads to that junction are not accessible and another junction is measured then it cannot be assured that the
highest temperature on the chip will be measured. Direct measurement should be used in this case.
When making the test measurements indicated below, the package shall be considered to have achieved thermal
equilibrium when the measured temperature difference, junction to case, reaches approximately 99 percent of its final value.
The temperature difference at that time will change at a rate less than
d(T - T ) < 0.03 (T - T )
J C J C
dt t
where t is the time after application of a power dissipation increment. The total time required for stabilization will typically be
less than a minute.
3.2.2 Direct measurement of junction temperature for determination of RθJR . The junction temperature of the thermally
limiting element within the semiconductor chip can be measured directly using an infrared microradiometer. The cap or lid
shall first be removed from the package to expose the active chip or device. The cavity shall not be covered with any IR
transparent material unless the chip is extremely large and has an extremely poor heat conduction path to the chip carrier.
The location of the junction to be measured should be referenced to a coordinate system on the chip so it can be relocated
after coating the chip. The active area of the chip shall be coated uniformly with a thin layer (25-50 µm thick) of a known
high emissivity (∈ > 0.8), low thermal conductivity material such as black pigmented lacquer. The package shall then be
placed on a temperature controlled heat sink and the case or mounting surface temperature stabilized at the specified value.
The microelectronic device under test shall then be operated at its rated power dissipation, the infrared microscope
crosshairs focused on the junction and scanned back and forth slightly at that location to maximize the radiance
measurement. That radiance measurement and the chip carrier temperature shall then be recorded. The power to the test
package shall then be turned off and the chip carrier allowed to return to the specified case or mounting surface
temperature. The emissivity of the coating over the junction region shall then be measured and the radiance from the
operating junction region shall be converted to temperature using this emissivity value. (Note that this method assumes the
emissivity of the coating material does not change appreciably with temperature. This assumption shall be valid if the
results are to be accurate and repeatable.)
If the junction to be measured is not specified then the test shall proceed as above except that the IR microscope crosshairs
shall be scanned over the whole active area of the chip to find and maximize the radiance measurement at the highest
temperature junction region.
The minimum width or length of the junction area shall be greater than 5 times the half power diameter of the objective lens
and greater than 5 times the thickness of the coating on the chip surface if this method is used to measure T . For
J(Peak)
junction element diameters between 5 and 1 times the half power diameter of the IR microscope objective lens, some
average junction temperature T , where T < T < T , will be measured.
J(Avg) J(Region) J(Avg) J(Peak)
METHOD 1012.1
4 November 1980
3

The following data shall be recorded for this test condition:
a. Peak or average junction temperature, T or T .
J(Peak) J(Avg)
b. Case or mounting surface temperature (usually 60°C ±0.5°C T , T ).
C M
c. Power dissipation, P , in the package.
D(Package)
d. Reference temperature measuring point.
e. Mounting arrangement.
f. Half power "spot" size of the IR microscope.
g. Thickness of the emissivity control coating (for T measurements only).
J(Avg)
h. Minimum width or length of the junction measured (for T measurements only).
J(Avg)
3.2.3 Indirect measurements of junction temperature for the determination of RθJR . The purpose of the test is to measure
the thermal resistance of integrated circuits by using particular semiconductor elements on the chip to indicate the device
junction temperature.
In order to obtain a realistic estimate of the operating average junction temperature, T , the whole chip or chips in the
J(Avg)
package should be powered in order to provide the proper internal temperature distribution. For other purposes though (see
section 3.2.1), the junction element being sensed need only be powered. During measurement of the junction temperature
the chip heating current shall be switched off while the junction calibration current remains stable. It is assumed that the
calibration current will not affect the circuit operation; if so, then the calibration current must be switched on as the power is
switched off.
The temperature sensitive device parameter is used as an indicator of an average junction temperature of the
semiconductor element for calculations of thermal resistance. The measured junction temperature is indicative of the
temperature only in the immediate vicinity of the element used to sense the temperature. Thus, if the junction element being
sensed is also dissipating power with a uniform heating current distribution, then T ≈ T for that particular junction
J(Avg) J(Peak)
element. If the current distribution is not uniform then T is measured. If the junction element being sensed is in the
J(Avg )
immediate vicinity of the element dissipating power then T will be measured. The heating power does not have to be
J(Region)
switched off when T is measured.
J(Region)
The temperature sensitive electrical parameters generally used to indirectly measure the junction temperature are the
forward voltage of diodes, and the emitter-base and the collector-base voltages of bipolar transistors. Other appropriate
temperature sensitive parameters may be used for indirectly measuring junction temperature for fabrication technologies
that do not lend themselves to sensing the active junction voltages. For example, the substrate diode(s) in junction-isolated
monolithic integrated circuits can be used as the temperature sensitive parameter for measurements of T . In this
J(Region)
particular case though, the heating power has to be switched off at the same time that the substrate diode is forward biased.
METHOD 1012.1
4 November 1980
4

3.2.3.1 Switching techniques for measuring T . The following symbols shall apply for the purpose of these
J(Avg)
measurements:
I - - - - - - - - - - - - - - - - Measuring current in milliamperes.
M
V - - - - - - - - - - - - - - - Value of temperature-sensitive parameter in millivolts, measured at I , and
MD M
corresponding to the temperature of the junction heated by P .
D
T - - - - - - - - - - - - - - - Calibration temperature in °C, measured at the reference point.
MC
V - - - - - - - - - - - - - - - Value of temperature-sensitive parameter in millivolts, measured at I and specific
MC M
value of T .
MC
The measurement of T using junction forward voltage as the TSP is made in the following manner:
J(Avg)
Step 1 - Measurement of the temperature coefficient of the TSP (calibration).
The coefficient of the temperature sensitive parameter is generated by measuring the TSP as a function of the reference
point temperature, for a specified constant measuring current, I , and collector voltage, by externally heating the device
M
under test in an oven or on a temperature controlled heat sink. The reference point temperature range used during
calibration shall encompass the temperature range encountered in the power application test (see step 2). The measuring
current is generally chosen such that the TSP decreases linearly with increasing temperature over the range of interest and
that negligible internal heating occurs during the measuring interval. A measuring current ranging from 0.05 to 5 mA is
generally used, depending on the rating and operating conditions of the device under test, for measuring the TSP. The
value of the TSP temperature coefficient, V /T , for the particular measuring current and collector voltage used in the
MC MC
test, is calculated from the calibration curve, V versus T .
MC MC
Step 2 - Power application test.
The power application test is performed in two parts. For both portions of the test, the reference point temperature is held
constant at a preset value. The first measurement to be made is that of the temperature sensitive parameter, i.e., V ,
MC
under operating conditions with the measuring current, I , and the collector voltage used during the calibration procedure.
M
The microelectronic device under test shall then be operated with heating power (P ) intermittently applied at greater than or
D
equal to 99 percent duty factor. The temperature- sensitive parameter V shall be measured during the interval between
MD
heating pulses (<100 µs) with constant measuring current, I , and the collector voltage that was applied during the
M
calibration procedure (see step 1).
Because some semiconductor element cooling occurs between the time that the heating power is removed and the time that
the temperature-sensitive parameter is measured, V may have to be extrapolated back to the time where the heating
MD
power was terminated by using the following mathematical expression which is valid for the first 100 µs of cooling:
V (t = 0) = V + V - V
MD MD1 MD2 MD1
t 1/2
1
t 1/2 - t 1/2
1 2
Where:
V (t = 0) = TSP, in millivolts, extrapolated to the time at which
MD
the heating power is terminated,
t = Delay time, in microseconds, after heating power is terminated,
V = TSP, in millivolts, at time t = t , and
MD1 1
V = TSP, in millivolts, at time t = t < t .
MD2 2 1
METHOD 1012.1
4 November 1980
5

If V (t) versus t1/2 is plotted on linear graph paper for the first 100 µs of cooling, the generated curve will be a straight line
MD
except during the initial portion where nonthermal switching transients dominate. The time t is the minimum time at which
2
the TSP can be measured as determined from the linear portion of the V (t) versus t1/2 cooling curve. Time t should be at
MD 1
least equal to t + 25 µs but less than 100 µs. The delay time before the TSP can be measured ranges from 1 to 50 µs for
2
most microelectronic devices. This extrapolation procedure is valid for semiconductor (junction) sensing elements >0.2 mm
(8 mils) in diameter over the delay time range of interest (1 to 50 µs).
When the error in the calculated thermal resistance caused by using V instead of the extrapolated value V (t = 0)
MD2 MD
exceeds 5 percent, the extrapolated value of V shall be used for calculating the average junction temperature.
MD
The heating power, P , shall be chosen such that the calculated junction-to- reference point temperature difference as
D
measured at V is greater than or equal to 20°C. The values of V , V , and P are recorded during the power
MD2 MD MC D
application test.
The following data shall be recorded for these test conditions:
a. Temperature sensitive electrical parameters (V , V (emitter-only switching), V (emitter and collector
F EB EB
switching), V , V , or other appropriate TSP).
CB F(subst)
b. Average junction temperature, T , is calculated from the equation:
J(Avg)
∆V - 1
MC
T = T + (V - V ),
J(AVG) R MD MC
∆T
MC
where: T = T or T
R C M
c. Case or mounting surface temperature, T or T , (usually 60° ±0.5°C).
C M
d. Power dissipation, P where P = P or P .
D D D(Package) D(Element)
e. Mounting arrangement.
3.2.3.2 Typical test circuits for indirect measurements of T . The circuit on figure 1012-3 can be used to sense V ,
J(Avg) F
V (emitter-only switches), V (emitter and collector switching), and V . The circuit is configured for heating power to be
EB EB CB
applied only to the junction element being sensed P for illustration purposes only.
D(Element)
The circuit on figure 1012-3 is controlled by a clock pulse with a pulse width less than or equal to 100 µs and repetition rate
less than or equal to 66.7 Hz. When the voltage level of the clock pulse is zero, the transistor Q1 is off and transistor Q2 is
on, and the emitter current through the device under test (DUT) is the sum of the constant heating current and the constant
measuring current. Biasing transistor Q1 on, shunts the heating current to ground and effectively reverse biases the diode
D1. The sample-and-hold unit is triggered when the heating current is removed and is used to monitor the TSP of the device
under test. During calibration, switch S4 is open.
The circuit on figure 1012-4 can be used to sense the forward voltage of the substrate diode of a junction isolated integrated
circuit. In this test circuit the microelectronic device under test is represented by a single transistor operated in a
common-emitter configuration. The substrate diode D is shown connected between the collector (most positive
SUBST
terminal) and the emitter (most negative terminal) of the integrated circuit under test. The type of circuitry needed to
interrupt the heating power will depend on the complexity of the integrated circuit being tested.
The circuit on figure 1012-4 is controlled by a clock pulse with a pulse width less than or equal to 100 µs and repetition rate
less than or equal to 66.7 Hz. When the voltage level of the clock pulse is zero, transistor Q1 being off and transistor Q2 on,
the device under test is dissipating heating power. Biasing transistor Q1 on and Q2 off, interrupts the heating power and
forward biases the substrate diode. The sample-and-hold unit is triggered when the heating current is removed and is used
to monitor the substrate diode forward voltage. During calibration, switch S1 is open.
METHOD 1012.1
4 November 1980
6

3.3 Thermal response time, junction to specified reference point, t .
JR
3.3.1 General considerations. When a step function of power dissipation is applied to a semiconductor device, the
junction temperature does not rise as a step function, but rather as a complex exponential curve. An infrared
microradiometer or the electrical technique, in which a precalibrated temperature sensitive device parameter is used to
sense the junction temperature, shall be used to generate the microelectronic device thermal response time.
When using electrical techniques, in which the device heating power is removed before the TSP is sensed for measuring the
thermal response time, the cooling curve technique shall be used. The measurement of the cooling curve is performed by
heating the device to steady state, switching the power off, and monitoring the junction temperature as the device cools.
The cooling curve technique is based upon the assumption that the cooling response of a device is the conjugate of the
heating response.
3.3.2 Measurement of junction temperature as a function of time for the determination of t . The change in junction
JR
temperature as a function of time resulting from the application or removal of a step function of heating power dissipation in
the junction(s) shall be observed using an infrared microradiometer with a response time of less than 100 µs, or electrical
equipment with a response time of less than 100 µs and sufficient sensitivity to read a precalibrated temperature sensitive
electrical parameter of the junction. During this test the device reference point temperature, as specified, shall be held
constant, the step function of power dissipation shall be applied or removed, and the waveform of the junction temperature
response versus time shall be recorded from the time of power application or removal to the time when the junction
temperature reaches a stable value.
The following data shall be recorded for this test condition:
a. Temperature sensitive electrical parameter (see section 3.2.3).
b. Infrared microscope spatial resolution (see section 3.2.2).
c. Peak, average, or region junction temperature as a function of time (see section 3.2.2 or 3.2.3 for details).
d. Case or mounting surface temperature T or T (usually 60°C ±0.5°C).
C M
e. Power dissipation, P or P m in the package.
D(Package) D(Element)
f. Reference temperature measuring point.
g. Mounting arrangement.
3.3.3 Typical test circuits for measurement of junction temperature as a function of time. The circuits depicted in section
3.2.3 are also used for the measurement of junction temperature as a function of time. The clock pulse is varied to give the
required step of heating power and the TSP is monitored on a cathode ray oscilloscope. When an infrared microradiometer
is used, the measuring current and TSP sensing circuitry is disconnected.
METHOD 1012.1
4 November 1980
7

3.4 Calculations of RθJR and t
JR
.
3.4.1 Calculations of package thermal resistance. The thermal resistance of a microelectronic device can be calculated
when the peak junction, average junction, or region junction temperature, T , T , or T , respectively, has been
J(Peak) J(Avg) J(Region)
measured in accordance with procedures outlined in sections 3.1 and 3.2. If the total package capability is to be assessed,
then rated power P should be applied to the device under test. For quality control purposes the power dissipation in
D(Packages)
the single test junction P can be used in the calculation of thermal resistance.
H(Element)
With the data recorded from each test, the thermal resistance shall be determined from:
RθJC(PEAK) = T
J(PEAK)
- T
C
, junction peak-to-case;
P
D(Package)
RθJC(Avg) = T
J(Avg)
- T
C
, junction average-to-case; or
P
D(Package)
RθJC(Region) = T
J(Region)
- T
C
, junction region-to-case;
P
D(Package)
For calculations of the junction element thermal resistance, P should be used in the previous equations. Note that
D(Element)
these thermal resistance values are independent of the heat sinking technique for the package. This is possible because
the case or chip carrier (reference) temperature is measured on the package itself in an accessible location which provides
a representative temperature in the major path of heat flow from the chip to the heat sink via the package.
3.4.2 Calculation of package thermal response time. The thermal response time of a microelectronic device can be
calculated when the peak junction, average junction, or region junction temperature, T , T , or T , respectively,
J(Peak) J(Avg) J(Region)
has been measured as a function of time in accordance with procedures outlined in section 3.3. If the total package
capability is to be assessed, then rated power P should be applied to the device under test. For quality control
D(Package)
purposes the power dissipation in the single test junction P can be used in the calculation of thermal response time.
D(Element)
With the data recorded from each test, the thermal response time shall be determined from a curve of junction temperature
versus time from the time of application or removal of the heating power to the time when the junction temperature reaches a
stable value. The thermal response time is 0.9 of this difference.
4. SUMMARY. The following details shall be specified in the applicable acquisition document:
a. Description of package; including number of chips, location of case or chip carrier temperature measurement(s),
and heat sinking arrangement.
b. Test condition(s), as applicable (see section 3).
c. Test voltage(s), current(s) and power dissipation of each chip.
d. Recorded data for each test condition, as applicable.
e. Symbol(s) with subscript designation(s) of the thermal characteristics determined to verify specified values of
these characteristics, as applicable.
f. Accept or reject criteria.
METHOD 1012.1
4 November 1980
8

FIGURE 1012-1. Temperature controlled heat sink.
METHOD 1012.1
4 November 1980
9

FIGURE 1012-1. Temperature controlled heat sink - Continued.
METHOD 1012.1
4 November 1980
10

FIGURE 1012-2. Temperature arrangement for mounting surface temperature measurements.
METHOD 1012.1
4 November 1980
11

TSP: Diode V - Switch S1 in position 2
F
Switch S2 in position 1
Transistor V (Emitter-only switching) - Switch S1 in position 2
EB
Switch S2 in position 2
Switch S3 in position 2
Transistor V (Emitter and collector switching) - Switch S1 in position 2
EB
Switch S2 in position 2
Switch S1 in position 1
Transistor V - Switch S1 in position 1
CB
Switch S2 in position 2
Switch S3 in position 1
FIGURE 1012-3. Typical test circuit for indirect measurement of T using
J(Avg)
p-n junction voltages of active devices.
METHOD 1012.1
4 November 1980
12

FIGURE 1012-4. Typical test circuit for indirect measurement of T using
J(Region)
the substrate diode of junction isolated integrated circuit.
METHOD 1012.1
4 November 1980
13

METHOD 1012.1
4 November 1980
14

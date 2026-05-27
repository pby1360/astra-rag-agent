---
standard: MIL-STD-810H
method: "522.2"
category: shock
language: en
---

# MIL-STD-810H Method 522.2 — BALLISTIC SHOCK

METHOD 522.2
METHOD 522.2
BALLISTIC SHOCK
CONTENTS
Paragraph Page
1. SCOPE ........................................................................................................................................................... 1
1.1 PURPOSE .......................................................................................................................................................... 1
1.2 APPLICATION ................................................................................................................................................... 1
1.2.1 BALLISTIC SHOCK DEFINITION ......................................................................................................................... 1
1.2.2 BALLISTIC SHOCK - MOMENTUM EXCHANGE .................................................................................................. 1
1.2.3 BALLISTIC SHOCK - PHYSICAL PHENOMENON .................................................................................................. 2
1.3 LIMITATIONS .................................................................................................................................................... 2
2. TAILORING GUIDANCE ........................................................................................................................... 3
2.1 SELECTING THE BALLISTIC SHOCK METHOD ................................................................................................... 3
2.1.1 EFFECTS OF BALLISTIC SHOCK ......................................................................................................................... 3
2.1.2 SEQUENCE AMONG OTHER METHODS.............................................................................................................. 3
2.2 SELECTING A PROCEDURE ................................................................................................................................ 3
2.2.1 PROCEDURE SELECTION CONSIDERATIONS ...................................................................................................... 5
2.2.2 DIFFERENCE AMONG PROCEDURES .................................................................................................................. 6
2.2.2.1 PROCEDURE I - BH&T ..................................................................................................................................... 6
2.2.2.2 PROCEDURE II - LSBSS ................................................................................................................................... 6
2.2.2.3 PROCEDURE III - LWSM .................................................................................................................................. 6
2.2.2.4 PROCEDURE IV - MECHANICAL SHOCK SIMULATOR ........................................................................................ 6
2.2.2.5 PROCEDURE V - MWSM .................................................................................................................................. 6
2.2.2.6 PROCEDURE VI - DROP TABLE ......................................................................................................................... 6
2.3 DETERMINE TEST LEVELS AND CONDITIONS ................................................................................................... 6
2.3.1 GENERAL CONSIDERATIONS - TERMINOLOGY .................................................................................................. 7
2.3.2 TEST CONDITIONS - SHOCK SPECTRUM TRANSIENT DURATION AND SCALING ................................................ 7
2.3.2.1 MEASURED DATA AVAILABLE FROM BALLISTIC SHOCK ................................................................................. 7
2.3.2.2 MEASURED DATA NOT AVAILABLE FROM BALLISTIC SHOCK ......................................................................... 8
2.3.3 BALLISTIC SHOCK QUALIFICATION - PROCEDURE I.......................................................................................... 8
2.3.4 BALLISTIC SHOCK QUALIFICATION - PROCEDURES II THROUGH VI ................................................................. 8
2.4 TEST ITEM CONFIGURATION ............................................................................................................................ 9
3. INFORMATION REQUIRED ..................................................................................................................... 9
3.1 PRETEST ........................................................................................................................................................... 9
3.2 DURING TEST ................................................................................................................................................... 9
3.3 POST-TEST ....................................................................................................................................................... 9
4. TEST PROCESS ......................................................................................................................................... 10
4.1 TEST FACILITY ............................................................................................................................................... 10
4.2 CONTROLS ..................................................................................................................................................... 10
4.3 TEST INTERRUPTION ...................................................................................................................................... 11
4.3.1 INTERRUPTION DUE TO TEST EQUIPMENT MALFUNCTION ............................................................................. 11
4.3.2 INTERRUPTION DUE TO TEST ITEM OPERATION FAILURE .............................................................................. 11
4.4 INSTRUMENTATION ........................................................................................................................................ 11
4.4.1 BALLISTIC SHOCK MEASUREMENT TRANSDUCERS ........................................................................................ 11
4.4.2 DATA ACQUISITION INSTRUMENTATION ........................................................................................................ 12
4.4.2.1 FILTERING AND FREQUENCY RESPONSE ......................................................................................................... 12
522.2-i

METHOD 522.2
CONTENTS - Continued
Paragraph Page
4.4.2.2 SLEW RATE .................................................................................................................................................... 14
4.4.2.3 HEADROOM .................................................................................................................................................... 14
4.5 DATA ANALYSIS ............................................................................................................................................ 15
4.6 TEST EXECUTION ........................................................................................................................................... 15
4.6.1 PREPARATION FOR TEST................................................................................................................................. 15
4.6.1.1 PRELIMINARY STEPS ...................................................................................................................................... 15
4.6.1.2 PRETEST CHECKOUT ...................................................................................................................................... 15
4.6.2 PROCEDURES .................................................................................................................................................. 16
4.6.2.1 PROCEDURE I - BH&T ................................................................................................................................... 16
4.6.2.2 PROCEDURE II - LSBSS ................................................................................................................................. 16
4.6.2.3 PROCEDURE III - LWSM ................................................................................................................................ 16
4.6.2.4 PROCEDURE IV - MECHANICAL SHOCK SIMULATOR ...................................................................................... 17
4.6.2.5 PROCEDURE V - MWSM ................................................................................................................................ 17
4.6.2.6 PROCEDURE VI - DROP TABLE ....................................................................................................................... 18
5. ANALYSIS OF RESULTS ......................................................................................................................... 18
6. REFERENCE/RELATED DOCUMENTS ............................................................................................... 18
6.1 REFERENCED DOCUMENTS ............................................................................................................................. 18
6.2 RELATED DOCUMENTS ................................................................................................................................... 18
TABLES
TABLE 522.2-I. BALLISTIC SHOCK CHARACTERISTICS ................................................................................................ 4
TABLE 522.2-II. SRS FUNCTION FOR SHOCK ................................................................................................................ 9
FIGURES
FIGURE 522.2 -1. SHOCK RESPONSE SPECTRA OF “DEFAULT” BALLISTIC SHOCK LIMITS (TABLES 522.2-I & II) ........... 4
FIGURE 522.2 -2. SHOCK RESPONSE SPECTRA FROM THREE DIFFERENT SENSORS NEEDED TO MEASURE THE ENTIRE
SPECTRUM (5 HZ TO 100,000 HZ) OF A BALLISTIC SHOCK EVENT ................................................... 12
FIGURE 522.2 -3. FILTER ATTENUATION (CONCEPTUAL, NOT FILTER SPECIFIC) ......................................................... 13
FIGURE 522.2 -4. ILLUSTRATION OF SAMPLING RATES AND OUT OF BAND “FOLD OVER” FREQUENCIES FOR DATA
ACQUISITION SYSTEMS ................................................................................................................... 14
522.2-ii

METHOD 522.2
METHOD 522.2
BALLISTIC SHOCK
NOTE: Tailoring is essential. Select methods, procedures, and parameter levels based on the
tailoring process described in Part One, paragraph 4.2.2, and Annex C. Apply the general
guidelines for laboratory test methods described in Part One, paragraph 5 of this Standard.
1. SCOPE.
1.1 Purpose.
This Method includes a set of ballistic shock tests generally involving momentum exchange between two or more
bodies, or momentum exchange between a liquid or gas and a solid performed to:
a. Provide a degree of confidence that materiel can structurally and functionally withstand the infrequent shock
effects caused by high levels of momentum exchange on a structural configuration to which the materiel is
mounted.
b. Experimentally estimate the materiel's fragility level relative to ballistic shock in order that shock mitigation
procedures may be employed to protect the materiel’s structural and functional integrity.
1.2 Application.
1.2.1 Ballistic Shock Definition.
Ballistic shock is a high-level shock that generally results from the impact of projectiles or ordnance on armored
combat vehicles. Armored combat vehicles must survive the shocks resulting from large caliber non-perforating
projectile impacts, mine blasts, and overhead artillery attacks, while still retaining their combat mission capabilities.
Paragraph 6.1, reference a, discusses the relationship between various shock environments (ballistic shock,
transportation shock, rail impact shock, etc.) for armored combat vehicles. Actual shock levels vary with the type of
vehicle, the specific munition used, the impact location or proximity, and where on the vehicle the shock is measured.
There is no intent here to define the actual shock environment for specific vehicles. Furthermore, it should be noted
that the ballistic shock technology is still rather limited in its ability to define and quantify the actual shock
phenomenon. Even though considerable progress has been made in the development of measurement techniques,
currently used instrumentation (especially the shock sensing gages) is still bulky and cumbersome to use. The
development of analytical (computational) methods to determine shock levels, shock propagation, and mitigation is
lagging behind the measurement technology. The analytical methods under development and in use to date have not
evolved to the level where their results can be relied upon to the degree that the need for testing is eliminated. That
is, the prediction of response to ballistic shock is, in general, not possible except in the simplest configurations. When
an armored vehicle is subjected to a non-perforating large caliber munition impact or blast, the structure locally
experiences a force loading of very high intensity and of relatively short duration. Though the force loading is
localized, the entire vehicle is subjected to stress waves traveling over the surface and through the structure. In certain
cases, pyrotechnic shocks have been used in ballistic shock simulations. There are several caveats in such testing.
The characteristics of ballistic shock are outlined in the following paragraph.
1.2.2 Ballistic Shock - Momentum Exchange.
Ballistic shock usually exhibits momentum exchange between two bodies or between a fluid and a solid. It commonly
results in velocity change in the support materiel. Ballistic shock has a portion of its characterization below 100 Hz,
and the magnitude of the ballistic shock response at a given point reasonably far from the ballistic shock source is a
function of the size of the momentum exchange. Ballistic shock will contain material wave propagation characteristics
(perhaps substantially nonlinear) but, in general, the material is deformed and accompanied by structural damping
other than damping natural to the material. For ballistic shock, structural connections do not necessarily display great
attenuation since low frequency structural response is generally easily transmitted over joints. In processing ballistic
shock data, it is important to be able to detect anomalies. With regard to measurement technology, accelerometers,
strain gages, and shock sensing gages may be used (see paragraph 6.1, reference a). In laboratory situations, laser
velocimeters are useful. Ballistic shock resistance is not, in general, “designed” into the materiel. The occurrence of
522.2-1

METHOD 522.2
a ballistic shock and its general nature can only be determined empirically from past experience based on well-defined
scenarios. Ballistic shock response of materiel in the field is, in general, very unpredictable and not repeatable among
materiel.
1.2.3 Ballistic Shock - Physical Phenomenon.
Ballistic shock is a physical phenomenon characterized by the overall material and mechanical response at a structure
point from elastic or inelastic impact. Such impact may produce a very high rate of momentum exchange at a point,
over a small finite area or over a large area. The high rate of momentum exchange may be caused by collision of two
elastic bodies or a pressure wave applied over a surface. General characteristics of ballistic shock environments are
as follows:
a. Near-the-source stress waves in the structure caused by high material strain rates (nonlinear material region)
that propagate into the near field and beyond.
b. Combined low and high frequency (10 Hz – 1,000,000 Hz) and very broadband frequency input.
c. High acceleration (300 g’s – 1,000,000 g’s) with comparatively high structural velocity and displacement
response.
d. Short-time duration (<180 msec).
e. High residual structure displacement, velocity, and acceleration response (after the event).
f. Caused by (1) an inelastic collision of two elastic bodies, or (2) an extremely high fluid pressure applied for
a short period of time to an elastic body surface coupled directly into the structure, and with point source
input, i.e., input is either highly localized as in the case of collision, or area source input, i.e., widely dispersed
as in the case of a pressure wave.
g. Comparatively high structural driving point impedance (P/v, where P is the collision force or pressure, and v
is the structural velocity). At the source, the impedance could be substantially less if the material particle
velocity is high.
h. Measurement response time histories that are very highly random in nature, i.e., little repeatability and very
dependent on the configuration details.
i. Shock response at points on the structure is somewhat affected by structural discontinuities.
j. Structural response may be accompanied by heat generated by the inelastic impact or the fluid blast wave.
k. The nature of the structural response to ballistic shock does not suggest that the materiel or its components
may be easily classified as being in the “near field” or “far field” of the ballistic shock device. In general,
materiel close to the source experiences high accelerations at high frequencies, whereas materiel far from the
source will, in general, experience high acceleration at low frequencies as a result of the filtering of the
intervening structural configuration.
1.3 Limitations.
Because of the highly specialized nature of ballistic shock and the substantial sensitivity of ballistic shock to the
configuration, apply it only after giving careful consideration to information contained in paragraph 6.1, references a
and b.
a. This Method does not include special provisions for performing ballistic shock tests at high or low
temperatures. Perform tests at room ambient temperature unless otherwise specified or if there is reason to
believe either operational high temperature or low temperature may enhance the ballistic shock environment.
b. This Method does not address secondary effects such as blast, EMI, and thermal.
522.2-2

METHOD 522.2
2. TAILORING GUIDANCE.
2.1 Selecting the Ballistic Shock Method.
After examining requirements documents and applying the tailoring process in Part One of this Standard to determine
where ballistic shock effects are foreseen in the life cycle of the materiel, use the following to confirm the need for
this Method and to place it in sequence with other methods.
2.1.1 Effects of Ballistic Shock.
In general, ballistic shock has the potential for producing adverse effects on all electronic, mechanical, and electro-
mechanical systems. In general, the level of adverse effects increases with the level and duration of the ballistic shock,
and decreases with the distance from the source (point or points of impact) of the ballistic shock. Durations for ballistic
shock that produce material stress waves with wavelengths that correspond with the natural frequency wavelengths of
micro electronic components within various system components will enhance adverse effects. Durations for ballistic
shock that produce structure response movement that correspond with the low frequency resonances of mechanical
and electro-mechanical systems will enhance the adverse effects. Examples of problems associated with ballistic
shock include:
a. System failure as a result of destruction of the structural integrity of micro electronic chips including their
mounting configuration.
b. System component failure as a result of relay chatter.
c. System component failure as a result of circuit card malfunction, circuit card damage, and electronic
connector failure. On occasion, circuit card contaminants having the potential to cause short circuits may be
dislodged under ballistic shock. Circuit card mounts may be subject to damage from substantial velocity
changes and large displacements.
d. Material failure as a result of cracks and fracture in crystals, ceramics, epoxies or glass envelopes.
e. System component failure as a result of sudden velocity change of the structural support of the system
component, or the internal structural configuration of the mechanical or electro-mechanical system.
2.1.2 Sequence Among Other Methods.
a. General. Use the anticipated life cycle sequence of events as a general sequence guide (see Part One,
paragraph 5.5).
b. Unique to this Method. Unless otherwise identified in the life cycle profile and, since ballistic shock is
normally experienced in combat and potentially near the end of the life cycle, normally schedule ballistic
shock tests late in the test sequence. In general, the ballistic shock tests can be considered independent of
the other tests because of their unique and specialized nature.
2.2 Selecting a Procedure.
This Method includes six ballistic shock test procedures. See paragraph 2.3.4 for the “default” approach to ballistic
shock testing when no field data are available.
a. Procedure I - Ballistic Hull and Turret (BH&T), Full Spectrum, Ballistic Shock Qualification. Replication
of the shock associated with ballistic impacts on armored vehicles can be accomplished by firing projectiles
at a “Ballistic Hull and Turret” (BH&T) with the materiel mounted inside. This procedure is very expensive
and requires that an actual vehicle or prototype be available, as well as appropriate threat munitions. Because
of these limitations, a variety of other approaches is often pursued. The variety of devices used to simulate
ballistic shock is described in paragraph 6.1, reference a.
b. Procedure II - Large Scale Ballistic Shock Simulator (LSBSS). Ballistic shock testing of complete
components over the entire spectrum (10 Hz to 100 kHz) defined in Table 522.2-I and in Figure 522.2-1 can
be accomplished using devices such as the Large Scale Ballistic Shock Simulator (LSBSS) described in
paragraph 6.1, reference a. This approach is used for components weighing up to 500 Kg (1100 lbs), and is
considerably less expensive than the BH&T approach of Procedure I.
c. Procedure III - Limited Spectrum, Light Weight Shock Machine (LWSM). Components weighing less than
113.6 kg (250 lb) and shock mounted to eliminate sensitivity to frequencies above 3 kHz can be tested over
522.2-3

METHOD 522.2
for other contractor machines). These machines produce a shock that lies within the envelope of the default
shock response spectrum described in paragraph 2.3.4 up to 10 kHz. Shock content is present above 10 kHz,
but it is not well defined. Use of a Mechanical Shock Simulator is less expensive than full spectrum
simulation, and may be appropriate for light weight items that are sensitive to shock up to 10 kHz.
e. Procedure V - Limited Spectrum, Medium Weight Shock Machine (MWSM). Components weighing less
than 2273 kg (5000 lb) and not sensitive to frequencies above 1 kHz can be tested over the spectrum from 10
Hz to 1 kHz of Table 522.2-I and Figure 522.2-1 using a MIL-S-901 Medium Weight Shock Machine
(MWSM) (paragraph 6.1, reference c) adjusted for 15 mm (0.59 in.) displacement limits. Use of the MWSW
may be appropriate for heavy components and subsystems that are shock mounted and/or are not sensitive to
high frequencies.
f. Procedure VI - Drop Table. Light weight components (typically less than 18 kg (40 lbs)) which are shock
mounted can often be evaluated for ballistic shock sensitivity at frequencies up to 500 Hz using a drop table.
This technique often results in overtest at the low frequencies. The vast majority of components that need
shock protection on an armored vehicle can be readily shock mounted. The commonly available drop test
machine is the least expensive and most accessible test technique. The shock table produces a half-sine
acceleration pulse that differs significantly from ballistic shock. The response of materiel on shock mounts
can be enveloped quite well with a half-sine acceleration pulse if an overtest at low frequencies and an
undertest at high frequencies is acceptable. Historically, these shortcomings have been acceptable for the
majority of ballistic shock qualification testing.
NOTES: Related Shock Tests:
1. High Impact / Shipboard Equipment. Perform shock tests for shipboard equipment in
accordance with MIL-S-901. The tests of MIL-S-901 are tailorable through the design of
the fixture that attaches the test item to the shock machine. Ensure the fixture is as similar
to the mounting method used in the actual use environment. High impact shocks for Army
armored combat vehicles should be tested using this Method.
2. Fuzes and Fuze Components. Perform shock tests for safety and operation of fuzes and fuze
components in accordance with MIL-STD-331 (paragraph 6.1, reference d).
3. Combined Temperature and Shock Tests. Perform shock tests at standard ambient
conditions (Part One, paragraph 5.1a) unless a high or low temperature shock test is required.
2.2.1 Procedure Selection Considerations.
Based on the test data requirements, determine which test procedure is applicable. In most cases, the selection of the
procedure will be dictated by the actual materiel configuration, carefully noting any gross structural discontinuities
that may serve to mitigate the effects of the ballistic shock on the materiel. In some cases, the selection of the
procedure will be driven by test practicality. Consider all ballistic shock environments anticipated for the materiel
during its life cycle, both in its logistic and operational modes. When selecting procedures, consider:
a. The Operational Purpose of the Materiel. From the requirements documents, determine the functions to be
performed by the materiel either during or after exposure to the ballistic shock environment.
b. The Natural Exposure Circumstances for Ballistic Shock. The natural exposure circumstances for ballistic
shock are based on well-selected scenarios from past experience and the chances of the occurrence of such
scenarios. For example, if an armored vehicle is subject to a mine blast, a number of assumptions must be
made in order to select an appropriate test for the ballistic shock procedure. In particular, the size of the
mine, the location of major pressure wave impact, the location of the materiel relative to the impact “point,”
etc. If the armored vehicle is subject to non-penetrating projectile impact, the energy input configuration will
be different from that of the mine, as will be the effects of the ballistic shock on the materiel within the
armored vehicle. In any case, condition each scenario to estimate the materiel response as a function of
amplitude level and frequency content. It will then be necessary to decide to which scenarios to test and
which testing is most critical. Some scenario responses may “envelope” others, which may reduce the need
522.2-5

METHOD 522.2
for certain testing such as road, rail, gunfiring, etc. In test planning, do not break up any measured or
predicted response to ballistic shock into separate amplitude and/or frequency ranges using different tests to
satisfy one procedure.
c. Required Data. The test data required to determine whether the operational purpose of the materiel has been
met.
d. Procedure Sequence. Refer to paragraph 2.1.2.
2.2.2 Difference Among Procedures.
2.2.2.1 Procedure I - BH&T.
Ballistic shock is applied in its natural form using live fire testing. Test items are mounted in the BH&T that replicates
the full-size vehicle in its “as designed” configuration and location. If required, “upweight” the vehicle to achieve
proper dynamic response. Appropriate threats (type, distance, orientation) are successively fired at the hull and/or
turret. This procedure is used to evaluate the operation of actual components, or the interaction between various
components during actual ballistic impacts. Also, this procedure is used to determine actual shock levels for one
particular engagement, that may be above or below the ‘default’ shock level specified in Table 522.2-I.
2.2.2.2 Procedure II - LSBSS.
LSBSS is a low cost option for producing the spectrum of ballistic shock without the expense of live fire testing. This
procedure is used primarily to test large, hard mounted components at the ‘default’ shock level specified in Table
522.2-I. It produces shock over the entire spectrum (10 Hz to over 100,000 Hz), and is useful in evaluating components
of unknown shock sensitivity.
2.2.2.3 Procedure III - LWSM.
Ballistic shock is simulated using a hammer impact. The test item is mounted on an anvil table of the shock machine
using the test item’s tactical mount. The anvil table receives the direct hammer impact that replicates the lower
frequencies of general threats to a hull or turret. This procedure is used to test shock mounted components (up to
113.6 kg (250 lb)), which are known to be insensitive to the higher frequency content of ballistic shock. This procedure
produces ‘partial spectrum’ testing (up to 3,000 Hz) at the ‘default’ level specified in Table 522.2-I.
2.2.2.4 Procedure IV - Mechanical Shock Simulator.
Ballistic shock is simulated using a metal-to-metal impact (gas driven projectile). The test item is mounted on a plate
of the shock machine using the test item’s tactical mount. This procedure is used to test small components (1.8 kg (4
lb)) for the smallest machine; higher weight for other contractor machines), that are known to be insensitive to the
highest frequency content of ballistic shock. This procedure produces ‘partial spectrum’ testing (up to 10,000 Hz) at
the ‘default’ level specified in Table 522.2-I.
2.2.2.5 Procedure V - MWSM.
Ballistic shock is simulated using a hammer impact. The test item is mounted on the anvil table of the shock machine
using the test item’s tactical mount. The anvil table receives the direct hammer impact, which replicates the lower
frequencies of general threats to a hull or turret. This procedure is used to test components up to 2273 kg (5000 lb) in
weight which are known to be insensitive to the higher frequencies of ballistic shock. This procedure produces ‘partial
spectrum’ testing (up to 1,000 Hz.) at the ‘default’ level specified in Table 522.2-I.
2.2.2.6 Procedure VI - Drop Table.
Ballistic shock is simulated by the impact resulting from a drop. The test item is mounted on the table of a commercial
drop machine using the test item’s tactical mounts. The table and test item are dropped from a calculated height. The
table receives the direct blow at the impact surface, which approximates the lower frequencies of general threat to a
hull or turret. This procedure is used for ‘partial spectrum’ testing of shock mounted components that can withstand
an overtest at low frequencies.
2.3 Determine Test Levels and Conditions.
Having selected one of the six ballistic shock procedures (based on the materiel’s requirements documents and the
tailoring process), complete the tailoring process by identifying appropriate parameter levels, applicable test
conditions and applicable test techniques for that procedure. Exercise extreme care in consideration of the details in
522.2-6

METHOD 522.2
the tailoring process. Base these selections on the requirements documents, the Life Cycle Environmental Profile
(LCEP), and information provided with this method. Consider the following basic information when selecting test
levels.
2.3.1 General Considerations - Terminology.
In general, response acceleration will be the experimental variable of measurement for ballistic shock. However, this
does not preclude other variables of measurement such as velocity, displacement, or strain from being measured and
processed in an analogous manner, as long as the interpretation, capabilities, and limitations of the measurement
variable are clear. Pay particular attention to the high frequency environment generated by the ballistic attack, as well
as the capabilities of the measurement system to accurately record the materiel’s responses. For the purpose of this
method, the terms that follow will be helpful in the discussion relative to analysis of response measurements from
ballistic shock testing.
a. Effective transient duration: The "effective transient duration" is the minimum length of time which contains
all significant amplitude time history magnitudes beginning at the noise floor of the instrumentation system
just prior to the initial pulse, and proceeding to the point that the amplitude time history is a combination of
measurement noise and substantially decayed structural response. In general, an experienced analyst is
required to determine the pertinent measurement duration to define the ballistic shock event. The longer the
duration of the ballistic shock, the more low frequency information is preserved. The amplitude time history
magnitude may be decomposed into several “shocks” with different effective transient durations if it appears
that the overall time history trace contains several independent “shock-like” events in which there are decay
to near noise floor of the instrumentation system between events. Each event may be considered a separate
shock.
b. Shock response spectrum analysis: Paragraph 6.1, reference e, defines the equivalent static acceleration
maximax Shock Response Spectrum (SRS) and provides examples of SRS computed for classical pulses.
The SRS value at a given undamped natural oscillator frequency, f , is defined to be the absolute value of the
n
maximum of the positive and negative acceleration responses of a mass for a given base input to a damped
single degree of freedom system. The base input is the measured shock amplitude time history over a
specified duration (the specified duration should be the effective transient duration). To some extent, for
processing of ballistic shock response data, the equivalent static acceleration maximax SRS has become the
primary analysis descriptor. In this measurement description, the maximax equivalent static acceleration
values are plotted on the ordinate with the undamped natural frequency of the single degree of freedom
system with base input plotted along the abscissa. Interpret the phrase “equivalent static acceleration”
literally only for rigid lightweight components on isolation mounts.
2.3.2 Test Conditions – Shock Spectrum Transient Duration and Scaling.
Derive the SRS and the effective transient duration, T, from measurements of the materiel’s response to a ballistic
shock environment or, if available, from dynamically scaled measurements of a similar environment. Because of the
inherent very high degree of randomness associated with the response to a ballistic shock, extreme care must be
exercised in dynamically scaling a similar environment. For ballistic shock, there are no known scaling laws because
of the sensitivity of the response to the size of the shock and the general configuration.
2.3.2.1 Measured Data Available From Ballistic Shock.
a. If measured data are available, the data may be processed utilizing the SRS. (The use of Fourier Spectra (FS)
or the Energy Spectral Density (ESD) is not recommended, but may be of interest in special cases.) For
engineering and historical purposes, the SRS has become the standard for measured data processing. In the
discussion to follow, it will be assumed that the SRS is the processing tool. In general, the maximax SRS
spectrum (equivalent static acceleration) is the main quantity of interest. With this background, determine
the shock response spectrum required for the test from analysis of the measured environmental acceleration
time history. After carefully qualifying the data, to make certain there are no anomalies in the amplitude
time histories, according to the recommendations provided in paragraph 6.1, reference f, compute the SRS.
The analyses will be performed for Q = 10 at a sequence of natural frequencies at intervals of at least 1/12th
octave spacing to span a frequency range consistent with the objective of the procedure.
b. Because sufficient field data are rarely available for statistical analysis, an increase over the envelope of the
available spectral data is sometimes used to establish the required test spectrum to account for variability of
522.2-7

METHOD 522.2
the environment. The degree of increase is based upon engineering judgment and should be supported by
rationale for that judgment. In these cases, it is often convenient to envelope the SRS by computing the
maximax spectra over the sample spectra and proceed to add a +6 dB margin to the SRS maximax envelope.
NOTE: This approach does not apply to the default values in Table 522.2-I.
2.3.2.2 Measured Data Not Available From Ballistic Shock.
If a data base is not available for a particular configuration, use (carefully) configuration similarity and any associated
measured data for prescribing a ballistic shock test. Because of the sensitivity of the ballistic shock to the system
configuration and the wide variability inherent in ballistic shock measurements, use caution in determining levels.
Table 522.2-I and Figure 522.2-1 give ‘default’ values for expected ballistic shock levels when no field measurement
results are available.
2.3.3 Ballistic Shock Qualification – Procedure I.
Ballistic Shock Qualification - Procedure I is different from the other ballistic shock methods in that the shock levels
are unknown until each particular shot (threat munition, attack angle, impact point, armor configuration, etc.) has been
fired and measurements have been made. The shock levels are determined by the interaction of the threat munition
and the armor as well as by the structure of the vehicle. Although the levels cannot be specified in advance, this
technique produces the most realistic shock levels.
2.3.4 Ballistic Shock Qualification – Procedures II Through VI.
For Ballistic Shock Procedures II through VI, subject the test item to the appropriate ballistic shock level a minimum
of three times in the axis of orientation of greatest shock sensitivity (i.e., the worst direction). Perform an operational
verification of the component during/after each test. For frequencies above 1 kHz, many ballistic shock events produce
similar shock levels in all three axes. If shock levels are known from previous measurements, the shock testing can
be tailored appropriately. If shock measurements are not available, use Steps a-g as outlined below.
a. Ensure the test item remains in place and that it continues to operate during and following shocks that are at
or below the average shock level specified in Table 522.2-I. The test item must also remain in place and
continue to operate following shocks that are at or below the worst case shock level specified in Table 522.2-
I. Ensure materiel critical to crew survival (e.g., fire suppression systems) continues to operate during and
following the worst case shock.
b. Mount the transducers used to measure the shock on the structure as near as possible to the structure mount.
Take triaxial measurements at this location. If triaxial measurements are not practical, make as many uniaxial
measurements as is practical.
c. Analyze the shock measurements in the time domain, as well as the frequency domain. Calculate the SRS
using a damping ratio of 5 percent of critical damping (Q = 10); calculate the SRS using at least 12 frequencies
per octave, proportionally spaced in the region from 10 Hz to 10 kHz (e.g., 120 frequencies spaced at
approximately 10, 10.59, 11.22, 11.89, 12.59, …, 8414, 8913, 9441, 10,000 Hz).
d. For a test shock to be considered an acceptable simulation of the requirement, 90 percent of the points in the
region from 10 Hz to 10 kHz must fall within the bounds listed in Table 522.2-II.
e. If more than 10 percent of the SRS points in the region from 10 Hz to 10 kHz are above the upper bound, an
overtest has occurred. If more than 90 percent of the SRS points lie between the upper and lower bounds,
the desired qualification test has occurred. If none of the above occurs, and more than 10 percent of the
points are below the lower bound, an undertest has occurred.
f. If the test item or its mount fails, during a desired or an undertest, redesign the materiel and/or its mount to
correct the deficiency.
g. Retest the redesigned materiel and/or its mount following the above procedure.
522.2-8

METHOD 522.2
(2) Any data measurement anomalies, e.g., high instrumentation noise levels, loss of sensors or sensor
mounting as a result of testing, etc.
4. TEST PROCESS.
4.1 Test Facility.
Paragraph 6.1, reference a, describes four useful devices for ballistic shock testing. The most common is perhaps the
drop table shock test machine used for shock testing of small items. For larger items that are sensitive to high
frequency shock, higher frequency content and can only tolerate limited displacement, the Light Weight Shock
Machine (LWSM) and Medium-Weight Shock Machine (MWSM) specified in MIL-S-901 can be useful tools for
ballistic shock simulation. For large items, the Large Scale Ballistic Shock Simulator (LSBSS) uses an explosive
charge to drive a plate to which the materiel is mounted.
a. A BH&T device is the armor shell of a vehicle. It must contain the actual, fully functional, vehicle armor,
but may not have an operational engine, suspension, gun, tracks, etc. The number of functional components
and total weight of the BH&T device are adjusted to meet the requirements of each individual test effort.
b. The LSBSS is a 22,700 kg (25 ton) structure that uses high explosives and hydraulic pressure to simulate the
shock experienced by armored vehicle components and materiel (up to 500 kg (1100 lb)) caused by the
impact of enemy projectiles.
c. The MIL-S-901 Light Weight Shock Machine uses a 182 kg (400 lb) hammer to impact an anvil plate
containing the test item. Hammer drops of 0.3 m (1 foot), 0.9 m (3 feet), and 1.5 m (5 feet) are used from
two directions in three axes if the worst case axis is unknown. If the worst case axis is known and agreed, it
is only necessary to test in the worst case axis.
d. Mechanical shock simulators use a metal-to-metal impact (air or hydraulically driven projectile). The
projectile impact is tuned to replicate the shock content (up to 10,000 Hz) of the ‘default’ shock level in Table
522.2-I.
e. The MIL-S-901 Medium-Weight Shock Machine uses a 1360 kg (3000 lb) hammer to impact an anvil table
containing the test item. Hammer height is a function of the weight on the anvil table (test item and all
fixturing), and is specified in Table I of MIL-S-901.
f. Drop tables typically have a mounting surface for the test item on an ‘anvil’ that is dropped from a known
height. In some machines, the anvil is accelerated by an elastic rope, hydraulic, or pneumatic pressure to
reach the desired impact velocity. The duration and shape (half-sine or saw tooth) of the impact acceleration
pulse are determined by a ‘programmer’ (elastic pad or hydro-pneumatic device) that, in turn, determines the
frequency content of the shock.
4.2 Controls / Tolerance.
a. For shock-mounted components, it is often necessary to determine the transfer function of the shock
mounting system. Typically, a ‘dummy weight’ of the appropriate mass and center of gravity is mounted in
place of the test item and subjected to full level shocks. The input shock and test item responses are measured
to verify performance of the shock mounts. Once shock mount performance has been verified, evaluation of
an operational test item can begin.
b. Prior to subjecting the test item to the full level shock, a variety of ‘preparation’ shocks are typically
performed. For Procedure I (BH&T), a low level ‘instrumentation check’ round is normally fired prior to
shooting actual threat ammunition. A typical ‘instrumentation check’ round would be 113 g to 453.6 g (4 to
16 oz.) of explosive detonated 2.54 to 45.7cm (1 to 18 inches) from the outer armor surface, and would
usually produce no more than 10 percent of the shock expected from threat munition. For Procedure II
(LSBSS), a low-level instrumentation check shot is usually fired prior to full level testing. For Procedure III
(MIL-S-901 LWSM), the 1 foot hammer blow is normally used to check instrumentation, and any
measurement problems are resolved prior to 0.9 and 1.5m (3 foot and 5 foot) hammer drops. For Procedure
V (MIL-S-901 MWSM), use the ‘Group 1’ hammer height for the instrumentation check. A similar approach
is used on Procedure VI, whereby a low-level drop is used to check instrumentation before conducting the
full level shock.
522.2-10

METHOD 522.2
c. For calibration procedures, review the guidance provided in Part One paragraphs 5.3.2. For tolerance
procedures, refer to Section 517 Paragraph 4.2.2 and 4.2.1.1.
4.3 Test Interruption.
Test interruptions can result from two or more situations, one being from failure or malfunction of test equipment.
The second type of test interruption results from failure or malfunction of the test item itself during operational checks.
4.3.1 Interruption Due To Test Equipment Malfunction.
a. General. See Part One, paragraph 5.11, of this Standard.
b. Specific to this Method.
(1) Undertest interruption. If an unscheduled interruption occurs that causes the test conditions to fall
below allowable limits, reinitiate the test at the end of the last successfully completed cycle.
(2) Overtest interruptions. If the test item(s) is exposed to test conditions that exceed allowable limits,
conduct an appropriate physical examination of the test item and perform an operational check (when
practical) before testing is resumed. This is especially true where a safety condition could exist, such
as with munitions. If a safety condition is discovered, the preferable course of action is to terminate
the test and reinitiate testing with a new test item. If this is not done and test item failure occurs during
the remainder of the test, the test results may be considered invalid. If no problem has been
encountered during the operational checkout or the visual inspection, reestablish pre-interruption
conditions and continue from the point where the test tolerances were exceeded. See paragraph 4.3.2
for test item operational failure guidance.
4.3.2 Interruption Due To Test Item Operation Failure.
Failure of the test item(s) to function as required during operational checks presents a situation with several possible
options.
a. The preferable option is to replace the test item with a “new” one and restart from Step 1.
b. A second option is to replace / repair the failed or non-functioning component or assembly with one that
functions as intended, and restart the entire test from Step 1.
NOTE: When evaluating failure interruptions, consider prior testing on the same test
item and consequences of such.
4.4 Instrumentation.
Acceleration or velocity measurement techniques that have been validated in shock environments containing the high
level, high frequency shock that characterizes ballistic shock must be used. See paragraph 6.1, reference g, for details.
In general, ballistic shock measurements require the use of at least two different measurement technologies to cross
check each other for validity. In addition, the frequency spectrum of ballistic shock content is generally so wide (10
Hz to more than 100,000 Hz) that no single transducer can make valid measurements over the entire spectrum. This
broad time frequency environment provides a challenge to calibration of measurement sensors and any tolerances
provided in the test plan.
4.4.1 Ballistic Shock Measurement Transducers.
As mentioned in reference g of paragraph 6.1, multiple transducers are usually required to make valid measurements
over the entire spectrum of the ballistic shock environment (see references g and h). Figure 522.2-2 illustrates the
limited “useful frequency range” of three different transducers. Note that the ATC Velocity Coil has a noticeable
resonance at 70 Hz, but it agrees with the BOBKAT sensor from 300 Hz to 1,000 Hz, and provides useful data out to
1 MHz. The BOBKAT sensor indicates erroneous values below 30 Hz, and above 2 KHz, but agrees with the LOFFI
from 30 Hz to 150 Hz and agrees with the ATC Velocity Coil from 400 Hz to 1 KHz. The LOFFI sensor provides
useful data from 5 Hz to 150 Hz. The resonant frequency, damping ratio, and useful frequency range of each
522.2-11

METHOD 522.2
transducer should be taken into consideration and must be documented, so that transducer anomalies can be identified,
if present in the measurement data.
Figure 522.2-2. Shock Response Spectra from three different sensors needed to measure the entire spectrum
(5 Hz to 100,000 Hz) of a ballistic shock event.
Transducers used in ballistic shock applications must be evaluated in the ballistic shock environment (roughly 1 MHz,
roughly 1 million g, described in paragraph 1.2.3 above). Both field testing (using high explosives) and laboratory
testing (such as the TCU shock machine and laser vibrometer described in reference g) are required to qualify
transducers for use in a ballistic shock environment.
4.4.2 Data Acquisition Instrumentation.
4.4.2.1 Filtering and Frequency Response.
The data recording instrumentation shall have flat frequency response to at least 100 kHz for at least one channel at
each measurement location. Attenuation of 3 dB at 100 kHz is acceptable. The digitizing rate must be at least 2.5
times the filtering frequency. Note that when measurements of peak amplitude are used to qualify the shock level, a
sample rate of at least 10 times the filtering frequency (1 million samples per second) is required. Additional, lower
frequency measurement channels, at the same location may be used for lower frequency response measurements.
It is imperative that a responsibly designed system to reject aliasing is employed. Analog anti-alias filters must be in
place before the digitizer. The selected anti-alias filtering must have an attenuation of 50 dB or greater, and a pass
band flatness within one dB across the frequency bandwidth of interest for the measurement (see Figure 522.2-3).
Subsequent resampling e.g., for purposes of decimation, must be in accordance with standard practices and consistent
with the analog anti-alias configuration (e.g. digital anti-alias filters must be in place before subsequent decimations).
522.2-12

METHOD 522.2
c. Use of an analog detector at each stage of amplification, to insure that no signal “clipping” occurs prior to
filtering, serves as acceptable documentation as to where “Out of Band Energy” distortion did, or did not
occur.
d. Setting the full scale recording range to a factor of roughly 25X above the expected signal level (i.e. a
“Headroom” of 25X) serves as acceptable protection from internal clipping due to “Out of Band Energy”. If
the expected level was 2,000g, for example, the full scale range would be set to 50,000g. Hence a 50,000g
“Out of Band Energy” signal could be accommodated without clipping. Unfortunately, the expected “In
Band” signal level would only use 4% of the full scale capability of the recorder, compromising signal
fidelity. Note that use of “Post Filter Gain” (gain applied after the anti-alias filter has removed the “Out of
Band Energy”), reduces the amount of headroom required. In the previous example, the pre-filter gain would
still be set to provide a range of 50,000g, but additional gain after the filter could amplify the signal before
digitization, thereby increasing fidelity. The headroom of the post-filter gain would depend on knowledge of
the expected in-band signal and fidelity requirements. For situations where the expected level is not well
understood a post-filter gain overhead of 10X is recommended, or 20,000g in the example case.
4.5 Data Analysis.
Detailed analysis procedures for evaluation of the problems peculiar to ballistic shock measurement have not been
established. Many (but not all) of the techniques described in paragraph 6.1, reference f are appropriate.
4.6 Test Execution.
4.6.1 Preparation for Test.
4.6.1.1 Preliminary Steps.
Prior to initiating any testing, review pretest information in the test plan to determine test details (e.g., procedures, test
item configuration, ballistic shock levels, number of ballistic shocks):
a. Choose the appropriate test procedure.
b. If the ballistic shock is a calibrated test, determine the appropriate ballistic shock levels for the test prior to
calibration.
c. Ensure the ballistic shock signal conditioning and recording devices have adequate amplitude range and
frequency bandwidth. It may be difficult to estimate a peak signal and arrange the instrumentation
appropriately. In general there is no data recovery from a clipped signal. However, for over-ranged signal
conditioning, it is usually possible to acquire meaningful results for a signal 20 dB above the noise floor of
the measurement system. In some cases, redundant measurements may be appropriate - one measurement
being over-ranged and one measurement ranged at the best estimate for the peak signal. The frequency
bandwidth of most recording devices is usually readily available, but ensures that recording device input
filtering does not limit the signal frequency bandwidth.
4.6.1.2 Pretest Checkout.
All items require a pretest checkout at standard ambient conditions to provide baseline data. Conduct the checkout as
follows:
Step 1 Conduct a complete visual examination of the test item with special attention to any micro electronic
circuitry areas. Pay particular attention to its platform mounting configuration and potential stress
wave transmission paths.
Step 2 Document the results.
Step 3 Where applicable, install the test item in its test fixture.
Step 4 Conduct an operational checkout in accordance with the approved test plan, along with simple tests
for ensuring the measurement system is responding properly.
Step 5 Document the results for comparison with test data.
Step 6 If the test item operates satisfactorily, proceed to the first test. If not, resolve the problem and restart
at Step 1.
522.2-15

METHOD 522.2
Step 7 Remove the test item and proceed with the calibration.
4.6.2 Procedures.
The following procedures provide the basis for collecting the necessary information concerning the platform and test
item undergoing ballistic shock. Since one of four or more ballistic shock devices may be employed, the instructions
below must be consistent with the ballistic shock device selected.
4.6.2.1 Procedure I – BH&T.
Step 1 Select the test conditions and mount the test item in a Ballistic Hull and Turret (BH&T), that may
require ‘upweighting’ to achieve the proper dynamic response. (In general, there will be no
calibration when actual hardware is used in this procedure). Select measurement techniques that
have been validated in ballistic shock environments. See paragraph 6.1, reference g, for examples.
Step 2 Perform an operational check on the test item.
Step 3 Fire threat munitions at the BH&T and verify that the test item operates as required. Typically,
make shock measurements at the mounting location (‘input shock’) and on the test item (‘test item
response’).
Step 4 Record necessary data for comparison with pretest data.
Step 5 Photograph the test item as necessary to document damage.
Step 6 Perform an operational check on the test item. Record performance data. See paragraph 5 for
analysis of results.
4.6.2.2 Procedure II – LSBSS.
Step 1 Mount the test item to the LSBSS using the same mounting hardware as would be used in the actual
armored vehicle. Select the orientation of the test item with the intent of producing the largest shock
in the ‘worst case’ axis.
NOTE: A ‘dummy’ test item is typically mounted until measurements confirm that the proper
explosive ‘recipe’ (i.e., combination of explosive weight, stand-off distance, and hydraulic
displacement) has been determined to obtain the shock levels specified in Table 522.2-I and on
Figure 522.2-1. Then mount an operational test item to the LSBSS.
Step 2 Fire the LSBSS and verify the test item is operating as required before, during, and after the shot.
If the test item fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item
failure.
Step 3 Record initial data for comparison with post test data.
Step 4 Fire three test shots at the shock level specified in Table 522.2-I.
Step 5 Inspect the test item; photograph any noted damage, and record data for comparison with pretest
data.
4.6.2.3 Procedure III – LWSM.
Step 1 Modify the mounting for the anvil plate to restrict total travel (including dynamic plate deformation)
to 15 mm (0.59 inch). Mount the test item to the LWSM using the same mounting hardware as
would be used in an actual armored vehicle. Choose the orientation of the test item with the intent
of producing the largest shock in the ‘worst case’ axis.
Step 2 Perform a pretest checkout and record data for comparison with post test data.
NOTE: Typically, make shock measurements at the ‘input’ location to ensure the low frequency shock
levels specified in Table 522.2-I and in Figure 522.2-1 have been attained on the 1.5 m (5 foot) drop.
522.2-16

METHOD 522.2
Step 3 Perform a 0.3 m (1 foot) hammer drop followed by an operational check; record data. If the test
item fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Otherwise, proceed to Step 4.
Step 4 Perform a 0.9 m (3 foot) hammer drop followed by an operational check; record data. If the test
item fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Otherwise, go to Step 5.
Step 5 Perform a 1.5 m (5 foot) hammer drop followed by an operational check; record data. If the test
item fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Step 6 Repeat Step 5 two more times.
Step 7 If the worst case axis is unknown (see paragraph 4.1c), repeat Steps 2-6 for each direction of each
axis for a total of 18 five-foot hammer drops. See paragraph 5 for analysis of results.
4.6.2.4 Procedure IV – Mechanical Shock Simulator.
Step 1 Mount the test item to the Mechanical Shock Simulator using the same mounting hardware as would
be used in the actual armored vehicle. Select the orientation of the test item with the intent of
producing the largest shock in the ‘worst case’ axis.
Step 2 Launch the mechanical shock simulator projectile and verify the test item is functioning as required
before, during, and after the shot.
Step 3 Record initial data for comparison with post test data.
Step 4 Conduct three test shots at the shock level specified in Table 522.2-I.
Step 5 If the worst case axis is unknown (see paragraph 4.1c), repeat Steps 2-6 for each direction of each
axis, for a total of 18 projectile impacts.
Step 6 Inspect the test item; photograph any noted damage, and record data for comparison with pretest
data. Perform an operational check on the test item. Record performance data. See paragraph 5 for
analysis of results.
4.6.2.5 Procedure V – MWSM.
Step 1 Modify the supports for the anvil table (by shimming the 4 table lifts) to restrict table total travel
(including dynamic plate deformation) to 15 mm (0.59 inch).
Step 2 Mount the test item to the MWSM using the same mounting hardware as would be used in an actual
combat vehicle. Choose the orientation of the test item with the intent of producing the largest shock
in the ‘worst case’ axis (see Step 7 below).
Step 3 Perform a pretest checkout and record data for comparison with post test data.
NOTE: Typically, make shock measurements at the ‘input’ location to ensure that the low-
frequency shock levels specified in Table 522.2-I and on Figure 522.2-1 have been attained on the
‘Group III’ drop (from MIL-S-901).
Step 4 Perform a ‘Group I height’ hammer drop followed by an operational check; record data. If the test
item fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Step 5 Perform a ‘Group III height’ hammer drop followed by an operational check; record data.
Step 6 Repeat Step 5 two more times.
Step 7 If the worst case axis is unknown (see paragraph 4.1c), repeat Steps 2-6 for each direction of each
axis for a total of 18 hammer drops at the Group III height.
522.2-17

METHOD 522.2
4.6.2.6 Procedure VI – Drop Table.
Step 1 Calculate the expected response of a shock mounted test item (or measurements from field tests may
be used) and calculate a shock response spectra (SRS). Choose a half-sine acceleration pulse whose
SRS ‘envelopes’ the expected response of the shock mounted item. Note that this approach typically
results in an overtest at the lowest frequencies.
Step 2 Hard mount the test item to the drop table.
Step 3 Conduct an operational check and record data for comparison with post test data. If the test item
operates satisfactorily, proceed to Step 4. If not, resolve the problems and repeat this step.
Step 4 Test using the appropriate half-sine acceleration pulse three times in each direction of all three axes
(18 drops).
Step 5 Conduct a performance check and record data for comparison with pretest data. See paragraph 5
for analysis of results.
5. ANALYSIS OF RESULTS.
In addition to the guidance provided in Part One, paragraphs 5.14 and 5.17, and in Part One, Annex A, Tasks 405 and
406, the following information is provided to assist in the evaluation of the test results. Analyze any failure of a test
item to meet the requirements of the system specifications, and consider related information. Carefully evaluate any
failure in the structural configuration of the test item, e.g., mounts, that may not directly impact failure of the
functioning of the materiel but that would lead to failure in its service environment conditions.
6. REFERENCE/RELATED DOCUMENTS.
6.1 Referenced Documents.
a. Walton, W. Scott, “Ballistic Shock Simulation Techniques for Testing Armored Vehicle Components”,
Proceedings of the 64th, Shock and Vibration Symposium, Volume I, October 1993, pp. 237-246. Shock &
Vibration Exchange (SAVE), 1104 Arvon Road, Arvonia, VA 23004.
b. Walton, W. Scott and Joseph Bucci, “The Rationale for Shock Specification and Shock Testing of Armored
Ground Combat Vehicles”, Proceedings of the 65th Shock and Vibration Symposium, Volume I, October
1994, pp. 285-293. Shock & Vibration Exchange (SAVE), 1104 Arvon Road, Arvonia, VA 23004.
c. MIL-S-901, “Shock Tests, H.I. (High Impact), Shipboard Machinery, Equipment, and Systems,
Requirements for”.
d. MIL-STD-331, “Fuze and Fuze Components, Environmental and Performance Tests for”.
e. Kelly, Ronald D. and George Richman, “Principles and Techniques of Shock Data Analysis”, Shock &
Vibration Exchange (SAVE), 1104 Arvon Road, Arvonia, VA 23004.
f. Handbook for Dynamic Data Acquisition and Analysis, IEST-RD-DTE012.2, Institute of Environmental
Sciences and Technology, Arlington Place One, 2340 S. Arlington Heights Road, Suite 100, Arlington
Heights, IL 60005-4516; Institute of Environmental Sciences and Technology Website.
g. Walton, W. Scott, “Pyroshock Evaluation of Ballistic Shock Measurement Techniques”, Proceedings of the
62nd Shock and Vibration Symposium, Volume 2, pp. 422-431, October 1991. Shock & Vibration Exchange
(SAVE), 1104 Arvon Road, Arvonia, VA 23004.
h. Hepner, Brandon, Monahan, Christopher, and Walton, W. Scott, “Improved Mid-Frequency Measurement
of Ballistic Shock”, 81st Shock and Vibration Symposium, October 2010. Shock & Vibration Exchange
(SAVE), 1104 Arvon Road, Arvonia, VA 23004.
6.2 Related Documents.
a. Allied Environmental Conditions and Test Publication (AECTP) 400, Mechanical Environmental Tests
(under STANAG 4370), Method 422.
522.2-18

METHOD 522.2
b. Egbert, Herbert W. “The History and Rationale of MIL-STD-810 (Edition 2)”, January 2010; Institute of
Environmental Sciences and Technology, Arlington Place One, 2340 S. Arlington Heights Road, Suite 100,
Arlington Heights, IL 60005-4516.
(Copies of Department of Defense Specifications, Standards, and Handbooks, and International
Standardization Agreements are available online at https://assist.dlamil.
Requests for other defense-related technical publications may be directed to the Defense Technical Information Center
(DTIC), ATTN: DTIC-BR, Suite 0944, 8725 John J. Kingman Road, Fort Belvoir VA 22060-6218,
1-800-225-3842 (Assistance--selection 3, option 2), http://www.dtic.mil/dtic/; and the National Technical
Information Service (NTIS), Springfield VA 22161, 1-800-553-NTIS (6847), http://wwwntis.gov/.
522.2-19

METHOD 522.2
(This page is intentionally blank.)
522.2-20

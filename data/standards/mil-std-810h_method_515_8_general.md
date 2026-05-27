---
standard: MIL-STD-810H
method: "515.8"
category: general
language: en
---

# MIL-STD-810H Method 515.8 — ACOUSTIC NOISE

METHOD 515.8
METHOD 515.8
ACOUSTIC NOISE
CONTENTS
Paragraph Page
1. SCOPE ........................................................................................................................................................... 1
1.1 PURPOSE .......................................................................................................................................................... 1
1.2 APPLICATION ................................................................................................................................................... 1
1.3 LIMITATIONS .................................................................................................................................................... 1
2. TAILORING GUIDANCE ........................................................................................................................... 1
2.1 SELECTING THE ACOUSTIC NOISE METHOD ..................................................................................................... 1
2.1.1 EFFECTS OF THE ACOUSTIC NOISE ENVIRONMENT ........................................................................................... 1
2.1.2 SEQUENCE AMONG OTHER METHODS.............................................................................................................. 2
2.2 SELECTING PROCEDURES ................................................................................................................................. 2
2.2.1 PROCEDURE SELECTION CONSIDERATIONS ...................................................................................................... 2
2.2.2 DIFFERENCE AMONG PROCEDURES .................................................................................................................. 2
2.3 DETERMINE TEST LEVELS AND CONDITIONS ................................................................................................... 3
2.3.1 GENERAL ......................................................................................................................................................... 3
2.3.2 USE OF MEASURED AND RELATED DATA ......................................................................................................... 3
2.3.3 TYPES OF ACOUSTIC EXCITATION .................................................................................................................... 3
2.3.3.1 DIFFUSE FIELD ................................................................................................................................................. 3
2.3.3.2 GRAZING INCIDENCE ACOUSTIC NOISE ............................................................................................................ 4
2.3.3.3 CAVITY RESONANCE ........................................................................................................................................ 4
2.3.3.4 ADDITIONAL TECHNICAL GUIDANCE ............................................................................................................... 4
2.4 TEST ITEM CONFIGURATION ............................................................................................................................ 4
3. INFORMATION REQUIRED ..................................................................................................................... 4
3.1 PRETEST ........................................................................................................................................................... 5
3.2 DURING TEST ................................................................................................................................................... 5
3.3 POST-TEST ....................................................................................................................................................... 5
4. TEST PROCESS ........................................................................................................................................... 5
4.1 TEST FACILITY ................................................................................................................................................. 5
4.2 CONTROLS ....................................................................................................................................................... 6
4.2.1 CONTROL OPTIONS .......................................................................................................................................... 6
4.2.1.1 SINGLE POINT NOISE CONTROL ....................................................................................................................... 6
4.2.1.2 MULTIPLE POINT NOISE CONTROL ................................................................................................................... 6
4.2.1.3 VIBRATION RESPONSE CONTROL ..................................................................................................................... 6
4.2.2 CONTROL METHODS ........................................................................................................................................ 6
4.2.3 OVERALL ACCURACY OF CONTROL ................................................................................................................. 6
4.2.4 CALIBRATION AND TOLERANCE ....................................................................................................................... 6
4.3 TEST INTERRUPTION ........................................................................................................................................ 7
4.3.1 INTERRUPTION DUE TO LABORATORY EQUIPMENT MALFUNCTION ................................................................. 7
4.3.2 INTERRUPTION DUE TO TEST ITEM OPERATION FAILURE ................................................................................ 7
4.4 TEST SETUP ...................................................................................................................................................... 7
4.5 TEST EXECUTION ............................................................................................................................................. 7
4.5.1 PREPARATION FOR TEST................................................................................................................................... 7
515.8-i

METHOD 515.8
CONTENTS - Continued
Paragraph Page
4.5.1.1 PRELIMINARY STEPS ........................................................................................................................................ 7
4.5.1.2 PRETEST STANDARD AMBIENT CHECKOUT ...................................................................................................... 7
4.5.2 INSTALLATION OF THE TEST ITEM .................................................................................................................... 7
4.5.2.1 DIFFUSE FIELD ................................................................................................................................................. 7
4.5.2.2 GRAZING INCIDENCE ACOUSTIC NOISE ............................................................................................................ 8
4.5.2.3 CAVITY RESONANCE ACOUSTIC NOISE ............................................................................................................ 8
4.5.3 PROCEDURE I - DIFFUSE FIELD ACOUSTIC NOISE TESTING .............................................................................. 8
4.5.3.1 PROCEDURE Ia - UNIFORM INTENSITY ACOUSTIC NOISE TESTING ................................................................... 8
4.5.3.2 PROCEDURE Ib - DIRECT FIELD ACOUSTIC NOISE TESTING .............................................................................. 9
4.5.4 PROCEDURE II - GRAZING INCIDENCE ACOUSTIC NOISE TESTING.................................................................... 9
4.5.5 PROCEDURE III - CAVITY RESONANCE ACOUSTIC NOISE TESTING ................................................................ 10
5. ANALYSIS OF RESULTS ......................................................................................................................... 10
6. REFERENCE/RELATED DOCUMENTS ............................................................................................... 10
6.1 REFERENCED DOCUMENTS ............................................................................................................................. 10
6.2 RELATED DOCUMENTS ................................................................................................................................... 10
TABLE
TABLE 515.8-I. TEST TOLERANCES ......................................................................................................................... 6
METHOD 515.8 ANNEX A
GUIDANCE FOR INITIAL TEST SEVERITY
1. BROADBAND RANDOM AND INCIDENCE NOISE TESTING....................................................... A-1
1.1 OVERALL SOUND PRESSURE LEVEL (OASPL) ............................................................................................. A-1
1.2 TEST SPECTRUM ........................................................................................................................................... A-2
1.3 SIMULATION OF AERODYNAMIC TURBULENCE ............................................................................................ A-3
2. CAVITY RESONANCE TESTING ........................................................................................................ A-3
3. EXTERNAL STORES TESTING ........................................................................................................... A-4
3.1 TEST SPECTRUM ........................................................................................................................................... A-4
3.2 TEST PARAMETERS ...................................................................................................................................... A-5
ANNEX A FIGURES
FIGURE 515.8A-1. APPLIED TEST SPECTRUM ........................................................................................................... A-2
FIGURE 515.8A-2. TYPICAL STORE PROFILE ............................................................................................................ A-4
FIGURE 515.8A-3. ONE-THIRD OCTAVE BAND SPECTRUM FOR ASSEMBLED EXTERNALLY CARRIED AIRCRAFT
STORES ....................................................................................................................................... A-4
515.8-ii

METHOD 515.8
CONTENTS - Continued
Paragraph Page
ANNEX A TABLES
TABLE 515.8A-I. OVERALL SOUND PRESSURE LEVELS AND DURATIONS ................................................................ A-1
TABLE 515.8A-II. ONE-THIRD OCTAVE BAND LEVELS FOR FIGURE 515.8A-1 ......................................................... A-2
TABLE 515.8A-III. CAVITY RESONANCE TEST CONDITIONS (SEE PARAGRAPH 6.1, REFERENCE A)............................ A-3
TABLE 515.8A-IV. SUGGESTED ACOUSTIC TEST LEVELS FOR ASSEMBLED EXTERNALLY CARRIED AIRCRAFT
STORES ....................................................................................................................................... A-5
METHOD 515.8 ANNEX B
ADDITIONAL TECHNICAL GUIDANCE
1. REVERBERATION CHAMBERS .......................................................................................................... B-1
2. PROGRESSIVE WAVE TUBES ............................................................................................................. B-1
3. ACOUSTIC NOISE CHARACTERISTICS ........................................................................................... B-1
4. CONTROL STRATEGIES ...................................................................................................................... B-1
5. DEFINITIONS .......................................................................................................................................... B-2
5.1 SOUND PRESSURE LEVEL ............................................................................................................................. B-2
5.2 THIRD OCTAVE FILTERS .............................................................................................................................. B-2
6. DIRECT FIELD ACOUSTIC TEST CHARACTERISTICS ............................................................... B-2
515.8-iii

METHOD 515.8
(This page is intentionally blank.)
515.8-iv

METHOD 515.8
METHOD 515.8
ACOUSTIC NOISE
NOTE: Tailoring is essential. Select methods, procedures, and parameter levels based on the
tailoring process described in Part One, paragraph 4.2.2, and Annex C. Apply the general
guidelines for laboratory test methods described in Part One, paragraph 5 of this Standard.
1. SCOPE.
1.1 Purpose.
The acoustic noise test is performed to determine the adequacy of materiel to resist the specified acoustic environment
without unacceptable degradation of its functional performance and/or structural integrity.
1.2 Application.
This test is applicable to systems, sub-systems, and units, hereafter called materiel, that must function and/or survive
in a severe acoustic noise environment. This test is also applicable for materiel located where acoustic noise excitation
is used in combination with, or in preference to mechanical vibration excitation for the simulation of aerodynamic
turbulence (Method 523.4).
1.3 Limitations.
Technical limitations restrict production and control of laboratory acoustic environments. Therefore, laboratory
acoustic fields can be significantly different from many of the real fluctuating pressure loadings classed as "acoustic".
Consider these limitations when choosing a test type and test facility, as well as in interpreting test results. For
example, diffuse field acoustic noise (see paragraph 2.3.3.1) better represents acoustics in internal cavities where local
reflection and re-radiation from vibrating structures predominate. For external skins exposed to aerodynamic
turbulence or jet noise, grazing incidence acoustic noise (see paragraph 2.3.3.2) more closely represents flow/acoustic
wave propagation along skin surfaces.
2. TAILORING GUIDANCE.
2.1 Selecting the Acoustic Noise Method.
After examining the requirements documents and applying the tailoring process in Part One of this Standard to
determine where acoustic noise may be encountered in the life cycle of the materiel, use the following to confirm the
need for this Method and to place it in sequence with other methods.
2.1.1 Effects of the Acoustic Noise Method.
The acoustic noise environment is produced by any mechanical or electromechanical device capable of causing large
airborne pressure fluctuations. In general, these pressure fluctuations are of an entirely random nature over a large
amplitude range (5000 Pa to 87000 Pa) (0.73 psi to 12.6 psi), and over a broad frequency band extending from 10 Hz
to 10000 Hz. On occasion there may exist very high amplitude discrete frequency pressure fluctuations referred to as
‘tones’. When pressure fluctuations impact materiel, generally, a transfer of energy takes place between the energy
(in the form of fluctuating pressure) in the surrounding air to the strain energy in materiel. This transfer of energy
will result in vibration of the materiel, in which case the vibrating materiel may re-radiate pressure energy, absorb
energy in materiel damping, or transfer energy to components or cavities interior to the materiel. Because of the large
amplitude and broad frequency range of the fluctuating pressure, measurement of materiel response is important. The
following list is not intended to be all-inclusive, but it provides examples of problems that could occur when materiel
is exposed to an acoustic noise environment.
a. Wire chafing.
b. Component acoustic and vibratory fatigue.
c. Component connecting wire fracture.
515.8-1

METHOD 515.8
d. Cracking of printed circuit boards.
e. Failure of wave guide components.
f. Intermittent operation of electrical contacts.
g. Cracking of small panel areas and structural elements.
h. Optical misalignment.
i. Loosening of small particles that may become lodged in circuits and mechanisms.
j. Excessive electrical noise.
2.1.2 Sequence Among Other Methods.
a. General. Use the anticipated life cycle sequence of events as a general sequence guide (see Part One,
paragraph 5.5).
b. Unique to this Method. Like vibration, the effects of acoustically induced stresses may affect materiel
performance under other environmental conditions, such as temperature, humidity, pressure,
electromagnetic, etc. When it is required to evaluate the effects of acoustic noise together with other
environments, and when a combined test is impractical, expose a single test item to all relevant environmental
conditions in turn. Consider an order of application of the tests that is compatible with the Life Cycle
Environmental Profile (LCEP) and sequence guidance in the individual methods.
2.2 Selecting Procedures.
This Method includes three acoustic noise test procedures. Determine which of the following procedure(s) to be used.
a. Procedure I (Diffuse Field) Ia – Uniform Intensity Acoustic Noise, Ib - Direct Field Acoustic Noise.
b. Procedure II (Grazing Incidence Acoustic Noise)
c. Procedure III (Cavity Resonance Acoustic Noise).
2.2.1 Procedure Selection Considerations.
The choice of test procedure is governed by the in-service acoustic environments and test purpose. Identify these
environments from consideration of the Life Cycle Environmental Profile (LCEP) as described in Part One, Annex A,
Task 402. When selecting procedures, consider:
a. The operational purpose of the materiel. From the requirements documents, determine the functions to be
performed by the materiel in an acoustic noise environment, the total lifetime exposure to acoustic noise, and
any limiting conditions.
b. The natural exposure circumstances.
c. The test data required to determine if the operational purpose (function and life) of the materiel has been met.
d. The procedure sequence within the acoustic noise method. If more than one of the enclosed procedures is to
be applied to the same test item, it is generally more appropriate to conduct the less damaging procedure first.
2.2.2 Difference Among Procedures.
While all procedures involve acoustic noise, they differ on the basis of how the acoustic noise fluctuating pressure is
generated and transferred to the materiel.
a. Procedure I - Diffuse Field
Ia - Uniform Intensity Acoustic Noise.
Procedure Ia has a uniform intensity shaped spectrum of acoustic noise that impacts all the exposed
materiel surfaces.
Ib - Direct Field Acoustic Noise (DFAN).
Procedure Ib uses normal incident plane waves in a shaped spectrum of acoustic noise to impact
directly on all exposed test article surfaces without external boundary reflections. Depending on
515.8-2

METHOD 515.8
the geometry of the test article this could produce magnitude variations on surfaces due to phasing
differences between the plane waves. In the case of large surface area, low mass density test articles
the phasing difference may excite primary structure modes in a different way than the diffuse
reverberant field. This fundamental difference and its impact on the structure must be weighed
against the advantages of the DFAN method. See annex B, paragraph 6 for more detailed
information.
b. Procedure II - Grazing Incidence Acoustic Noise. Procedure II includes a high intensity, rapidly fluctuating
acoustic noise with a shaped spectrum that impacts the materiel surfaces in a particular direction - generally
along the long dimension of the materiel.
c. Procedure III - Cavity Resonance Acoustic Noise. In Procedure III, the intensity and, to a great extent, the
frequency content of the acoustic noise spectrum is governed by the relationship between the geometrical
configuration of the cavity and the materiel within the cavity.
2.3 Determine Test Levels and Conditions.
2.3.1 General.
Having selected this Method and relevant procedures (based on the materiel’s requirements and the tailoring process),
it is necessary to complete the tailoring process by selecting specific parameter levels and special test
conditions/techniques for these procedures based on the requirements documents, Life Cycle Environmental Profile,
and information provided with this procedure. From these sources of information, determine test excitation parameters
and the functions to be performed by the materiel in acoustic noise environments or following exposure to these
environments.
2.3.2 Use of Measured and Related Data.
Wherever possible, use specifically measured data to develop the test excitation parameters and obtain a better
simulation of the actual environment. Obtain data at the materiel location, preferably on the specific platform or,
alternatively, on the same platform type. In general, the data will be a function of the intended form of simulation. In
some cases, only microphone sound pressure levels will be useful, and in other cases materiel acceleration response
measurements will be useful.
2.3.3 Types of Acoustic Excitation.
2.3.3.1 Diffuse Field
2.3.3.1.1 Uniform Intensity Acoustic Noise.
A diffuse field is generated in a reverberation chamber. Normally wide band random excitation is provided and the
spectrum is shaped. This test is applicable to materiel or structures that have to function or survive in an acoustic
noise field such as that produced by aerospace vehicles, power plants and other sources of high intensity acoustic
noise. Since this test provides an efficient means of inducing vibration above 100 Hz, the test may also be used to
complement a mechanical vibration test, using acoustic energy to induce mechanical responses in internally mounted
materiel. In this role, the test is applicable to items such as installed materiel in airborne stores carried externally on
high performance aircraft. However, since the excitation mechanism induced by a diffuse field is different from that
induced by aerodynamic turbulence, when used in this role, this test is not necessarily suitable for testing the structural
integrity of thin shell structures interfacing directly with the acoustic noise. A practical guideline is that acoustic tests
are not required if materiel is exposed to broadband random noise at a sound pressure level less than 130 dB (reference
20 µPascal) overall, and if its exposure in every one Hertz band is less than 100 dB (reference 20 µPascal). A diffuse
field acoustic test is usually defined by the following parameters:
a. Spectrum levels.
b. Frequency range.
c. Overall sound pressure level.
d. Duration of the test.
515.8-3

METHOD 515.8
2.3.3.1.2 Direct Field Acoustic Noise.
A direct field is generated by audio drivers arranged to encircle the test article. Two different control schemes can be
used to perform a direct field test. One method, known as single input, single output or SISO, uses a single drive signal
to all acoustic drivers with multiple control microphones averaged to produce the control measurement. This method
will produce a set of correlated plane waves that may combine to produce large magnitude variations creating local
fluctuations on the test article surface. Magnitude variations as much as +12dB can be experienced. This variation
may not be acceptable for some applications. The second method, known as Multiple Input, Multiple Output or MIMO,
uses multiple independent drive signals to control multiple independent microphone locations. This method produces
a more uncorrelated field that is much more uniform than the SISO field. Magnitude variations in the range of +3dB
are typical when using MIMO control. All other characteristics of diffuse field testing described in 2.3.3.1.1 also
apply to the direct field method.
2.3.3.2 Grazing Incidence Acoustic Noise.
Grazing incidence acoustic noise is generated in a duct, popularly known as a progressive wave tube. Normally, wide
band random noise with a shaped spectrum is directed along the duct. This test is applicable to assembled systems
that have to operate or survive in a service environment of pressure fluctuations over the surface, such as exist in
aerodynamic turbulence. These conditions are particularly relevant to aircraft panels, where aerodynamic turbulence
will exist on one side only, and to externally carried stores subjected to aerodynamic turbulence excitation over their
total external exposed surface. In the case of a panel, the test item will be mounted in the wall of the duct so that
grazing incidence excitation is applied to one side only. An aircraft carried store such as a missile will be mounted
co-axially within the duct such that the excitation is applied over the whole of the external surface. A grazing incidence
acoustic noise test is usually defined by the following parameters:
a. Spectrum levels.
b. Frequency range.
c. Overall sound pressure level.
d. Duration of the test.
2.3.3.3 Cavity Resonance.
A resonance condition is generated when a cavity, such as that represented by an open weapons bay on an aircraft, is
excited by the airflow over it. This causes oscillation of the air within the cavity at frequencies dependent upon the
cavity dimensions and the aerodynamic flow conditions. In turn, this can induce vibration of the structure and of
components in and near the cavity. The resonance condition can be simulated by the application of a sinusoidal
acoustic source, tuned to the correct frequency of the open cavity. The resonance condition will occur when the
control microphone response reaches a maximum in a sound field held at a constant sound pressure level over the
frequency range. A cavity resonance test is defined by the following parameters:
a. Noise frequency.
b. Overall sound pressure level within the cavity.
c. Duration of the test.
2.3.3.4 Additional Technical Guidance.
Additional Guidance related to the various types of Acoustic Excitation is given in Annex B
2.4 Test Item Configuration.
(See Part One, paragraph 5.8.) Where relevant, function the test item, and measure and record performance data
during each test phase and/or each acoustic level applied.
3. INFORMATION REQUIRED.
The following information is necessary to properly conduct the acoustic test.
515.8-4

METHOD 515.8
3.1 Pretest.
a. General. See the information listed in Part One, paragraphs 5.7, 5.8, 5.9, 5.11 and 5.12; and Part One, Annex
A, Task 405 of this Standard.
b. Specific to this Method.
(1) Establish test levels and durations using projected Life Cycle Environmental Profiles, available data
or data acquired directly from an environmental data-gathering program. When these data are not
available, use the guidance on developing initial test severities in Annex A. Consider these overall
sound pressure levels (OASPL) (Annex A, Table 515.8A-I) as initial values until measured data are
obtained. The test selected may not necessarily be an adequate simulation of the complete
environment and consequently a supporting assessment may be necessary to complement the test
results.
(2) If the test item is required to operate during the test; the operating checks required are pretest, during
the test, and post test. For the pre- and post test checks, specify whether they are performed with the
test item installed in the test facility. Define the details required to perform the test, including the
method of attachment or suspension of the test item, the surfaces to be exposed, effect of gravity and
any consequent precautions. Identify the control and monitor points, or a procedure to select these
points. Define test interruption, test completion and failure criteria.
c. Tailoring. Necessary variations in the basic test procedures to accommodate LCEP requirements and/or
facility limitations.
3.2 During Test.
a. General. See the information listed in Part One, paragraph 5.10, and in Part One, Annex A, Tasks 405 and
406.
b. Specific to this Method.
(1) Collect outputs of microphones, test control averages, test item operating parameters and any other
relevant transducers at appropriate test times.
(2) Collect log/records of materiel operating parameters.
(3) Give particular attention to interactions of the input excitation (diffuse, directional or tonal).
(4) Record transient behavior in the input representing a test anomaly.
3.3 Post-Test.
The following post test data shall be included in the test report.
a. General. See the information listed in Part One, paragraph. 5.13; and in Part One, Annex A, Task 406 of this
Standard.
b. Specific to this Method.
(1) Identify any indication of failure under specified failure criteria. Account for tolerance excesses when
testing large materiel, the number of simultaneous test items in Procedure I, and any other
environmental conditions at which testing was carried out, if other than standard laboratory conditions.
(2) Ensure detailed data analysis for verification of the input to the test item, i.e., the acoustic field and
the response monitoring of the test item, are in accordance with the test plan.
(3) Any deviations from the test plan.
4. TEST PROCESS.
4.1 Test Facility.
Ensure the apparatus used to perform the acoustic test has sufficient capacity to adequately reproduce the input
requirements. Diffuse acoustic field apparatus that produce uniform acoustic fields above 165 dB are rare. For high
515.8-5

METHOD 515.8
4.3 Test Interruption.
Test interruptions can result from two or more situations, one being from failure or malfunction of test laboratory
equipment. The second type of test interruption results from failure or malfunction of the test item itself during
required or optional performance checks.
4.3.1 Interruption Due To Laboratory Equipment Malfunction.
a. General. See Part One, paragraph 5.11 of this Standard.
b. Specific to this Method. Interruption of an acoustic noise test is unlikely to generate any adverse effects.
Normally, continue the test from the point of interruption.
4.3.2 Interruption Due To Test Item Operation Failure.
Failure of the test item(s) to function as required during mandatory or optional performance checks during testing
presents a situation with several possible options.
a. The preferable option is to replace the test item with a “new” one and restart from Step 1.
b. A second option is to replace / repair the failed or non-functioning component or assembly with one that
functions as intended, and restart the entire test from Step 1.
NOTE: When evaluating failure interruptions, consider prior testing on the same test item and
any consequences of such.
4.4 Test Setup.
a. General. See Part One, paragraph 5.8.
b. Unique to this Method. Tests will normally be carried out with the test item mounted in the correct attitude,
unless it is shown that the performance of the test item is not affected by gravity.
4.5 Test Execution.
The following steps, alone or in combination, provide the basis for collecting necessary information concerning the
test item in an acoustic environment.
4.5.1 Preparation for Test.
4.5.1.1 Preliminary Steps.
Before starting the test, determine the test details (e.g., procedure variations, test item configuration, cycles, durations,
parameter levels for storage/operation, etc.) from the test plan. (See paragraph 3.1 above.)
4.5.1.2 Pretest Standard Ambient Checkout.
a. Unless otherwise specified, allow the test item to stabilize at standard ambient conditions.
b. Perform a physical inspection and operational checks before and after testing. Define the requirements for
these checks in the test plan. If these checks are required during the test sequence, specify the time intervals
at which they are required.
c. Ensure that all test environment monitoring instrumentation and test item function monitoring
instrumentation is consistent with the calibration and test tolerance procedures, and are generally consistent
with the guidance provided in Part One, paragraphs 5.3.2 and 5.2, respectively.
4.5.2 Installation of the Test item.
4.5.2.1 Diffuse Field
4.5.2.1.1 Uniform Intensity Acoustic Noise.
Suspend the test item (or as otherwise mounted) in a reverberation chamber on an elastic system in such a manner that
all appropriate external surfaces are exposed to the acoustic field and no surface is parallel to a chamber surface.
515.8-7

METHOD 515.8
Ensure the resonance frequency of the mounting system with the specimen is less than 25 Hz or 1/4 of the minimum
test frequency, whichever is less. If cables, pipes etc., are required to be connected to the test item during the test,
arrange them to provide similar restraint and mass as in service. Locate a microphone in proximity to each major
different face of the test item at a distance of 0.5 meter (1.64 ft) from the face, or midway between the center of the
face and the chamber wall, whichever is smaller. Average the outputs from these microphones to provide a single
control signal. When the chamber is provided with a single noise injection point, place one microphone between the
test item and the chamber wall furthest from the noise source. The orientation of the microphones in such a facility is
not critical, but do not set the microphone axes normal to any flat surface. Calibrate the microphones for random
incidence.
4.5.2.1.2 Direct Field Acoustic Noise (DFAN).
The test item should be surrounded by a circular array of acoustic drivers to a height of at least 1meter (3.28 ft) above
the test article. The arrangement should avoid symmetry to reduce the potential for adverse coupling of plane waves.
The test article can be mounted on a platform or suspended. Multiple microphones, eight to sixteen, should be used
for control with either the SISO or MIMO methods (see annex B, paragraph 6). The microphones should be placed
randomly around the test article. The distance from the surface of the drivers to the control microphones should be no
less than 1m (3.28 ft). The distance from the surface of the test article to the control microphones should also be no
less than 1m (3.28 ft) unless the pre-test characterization determines there are no structure induced pressure effects on
the microphone. The height of the control microphones should be centered at mid-height of the test item and randomly
varied up and down by about one-half of the test item height. The orientation of the free-field microphones in a DFAN
test arrangement is not critical. However, reflections from the test article can be minimized with the microphone
oriented toward the sound source with a 0 degree incidence (see Paragraph 6, reference c, figure 3.7). Most modern
day, quality measurement, free-field microphones are factory adjusted to compensate for incident angle. This
phenomenon is most pronounced at high frequencies, above 10kHz for a 1/4" microphone, and is inversely
proportional to microphone diaphragm diameter.
4.5.2.2 Grazing Incidence Acoustic Noise.
Mount test items such as panels in the wall of the duct such that the required test surfaces are exposed to the acoustic
excitation. Ensure this surface is flush with the inner surface of the duct to prevent the introduction of cavity resonance
or local turbulence effects. Suspend test items (such as aircraft external stores) centrally within the duct, on an elastic
support. Orient the test item such that the appropriate surfaces are subjected to progressive acoustic waves. For
example, orient an aircraft external store parallel to the duct centerline so that the acoustic waves sweep the length of
the store. Ensure the rigid body modes of the test item are lower than 25 Hz or 1/4 of the lowest test frequency,
whichever is less. Ensure that no spurious acoustic or vibratory inputs are introduced by the test support system or by
any ancillary structure. Mount the microphone(s) for control and monitoring of test conditions in the duct wall
opposite the test panel. Select other positions within the duct assuming the microphone is positioned so that it responds
to only grazing incidence waves, and that the necessary corrections are applied to the measured level. Calibrate the
microphones for grazing incidence.
4.5.2.3 Cavity Resonance Acoustic Noise.
Suspend the test item (or as otherwise mounted) in a reverberation chamber such that only that part of the cavity to be
tested is exposed to the direct application of acoustic energy. Protect all other surfaces so that their level of acoustic
excitation is reduced by 20 dB. Do not use protective coverings that provide any additional vibration damping to the
structure. Do not locate the microphone for control of the test within the cavity to be tested.
4.5.3 Procedure I – Diffuse Field Acoustic Noise Testing
4.5.3.1 Procedure Ia - Uniform Intensity Acoustic Noise Testing.
Step 1 Install the test item in the reverberation chamber in accordance with paragraph 4.5.2.1.1
Step 2 Select microphone positions for control, monitoring, and control strategy in accordance with
paragraph 4.5.2.1.1.
Step 3 When using open loop control, remove the test item and confirm the specified overall sound pressure
level and spectrum can be achieved in an empty chamber, then replace the test item in the chamber.
Step 4 Precondition the test item in accordance with paragraph 4.5.1.2.
515.8-8

METHOD 515.8
Step 5 Conduct initial checks in accordance with paragraph 4.5.1.2.
Step 6 Apply the test spectrum for the specified time. If required, carry out inspections and operational
checks in accordance with paragraph 4.5.1.2. If the test item fails to operate as intended, follow the
guidance in paragraph 4.3.2 for test item failure.
Step 7 Record the test acoustic field at each microphone, any average used in test control, and other
pertinent transducer outputs. Make the recordings at the beginning, midpoint, and end of each test
run. Where test runs are longer than one hour, record every one-half hour.
Step 8 Carry out the final inspection and operational checks, and see paragraph 5 for analysis of results.
Step 9 Remove the test item from the chamber.
Step 10 In all cases, record the information required.
4.5.3.2 Procedure Ib - Direct Field Acoustic Noise Testing.
Step 1 Build a test setup using a test item simulator in accordance with paragraph 4.5.2.1.2
Step 2 Select microphone positions for control, monitoring, and control strategy in accordance with
paragraph 4.5.2.1.2.
Step 3 Perform a pre-test using the simulator to confirm the specified overall sound pressure level and
spectrum can be achieved. Also verify any special control features to be used such as; abort
tolerances, response limits, field shaping and emergency shut-down procedures. Monitor the
resulting field for uniformity, coherence and structural response, if available. Then replace the
simulator with the actual test item in the speaker circle.
Step 4 Precondition the test item in accordance with paragraph 4.5.1.2.
Step 5 Conduct initial checks in accordance with paragraph 4.5.1.2.
Step 6 Apply the test spectrum for the specified time. Use multiple runs if the allowable audio system full
level ON (duty cycle as discussed in Annex B, paragraph 6) time is less than the total test time. If
required, carry out inspections and operational checks in accordance with paragraph 4.5.1.2. If the
test item fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Step 7 Record the test acoustic field at each microphone, any average used in test control, and other
pertinent transducer outputs. Make the recordings at the beginning, midpoint, and end of each test
run. Where test runs are longer than one hour, record every one-half hour.
Step 8 Carry out the final inspection and operational checks, and see paragraph 5 for analysis of results.
Step 9 Remove the test item from the circle.
Step 10 In all cases, record the information required.
4.5.4 Procedure II - Grazing Incidence Acoustic Noise Testing.
Step 1 Install the test item in accordance with paragraph 4.5.2.2.
Step 2 Select microphone positions for control, monitoring, and control strategy in accordance with
paragraph 4.5.2.2.
Step 3 Precondition the test item in accordance with paragraph 4.5.1.2.
Step 4 Conduct initial checks in accordance with paragraph 4.5.1.2.
Step 5 Apply the test spectrum for the specified time. If required, carry out inspections and operational
checks in accordance with paragraph 4.5.1.2. If the test item fails to operate as intended, follow the
guidance in paragraph 4.3.2 for test item failure.
Step 6 Record the test acoustic field at each microphone, any average used in test control, and other
pertinent transducer outputs. Make recordings at the beginning, end and midpoint of each test run.
Where test runs are longer than one hour, record every one-half hour.
515.8-9

METHOD 515.8
Step 7 Carry out the final inspection and operational checks, and see paragraph 5 for analysis of results.
Step 8 Remove the test item from the duct.
Step 9 In all cases, record the information required.
4.5.5 Procedure III - Cavity Resonance Acoustic Noise Testing.
Step 1 Install the test item into the chamber in accordance with paragraph 4.5.2.3.
Step 2 Locate the control microphone in accordance with paragraph 4.5.2.3.
Step 3 Precondition the test item in accordance with paragraph 4.5.1.2.
Step 4 Conduct initial checks in accordance with paragraph 4.5.1.2.
Step 5 Apply the sinusoidal acoustic excitation at the required frequencies (see Annex A, Table 515.8A-
II). Adjust the test parameters to the specified levels and apply for the specified time. If required,
carry out inspections and operational checks in accordance with paragraph 4.5.1.2. If the test item
fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Step 6 Record the test acoustic field at each microphone, any average used in test control, and other
pertinent transducer outputs. Make recordings at the beginning, midpoint, and end of each test run.
Where test runs are longer than one hour, record every one-half hour.
Step 7 Perform the final physical inspection and operational checks, and see paragraph 5 for analysis of
results.
Step 8 Remove the test item from the chamber.
Step 9 In all cases, record the information required.
5. ANALYSIS OF RESULTS.
Refer to Part One, paragraphs 5.14 and 5.17; and Part One, Annex A, Task 406.
6. REFERENCE/RELATED DOCUMENTS.
6.1 Referenced Documents.
a. AFFDL-TR-76-91; Volume II: "A Review of Methods for Estimation of Aero-acoustic Loads on Flight
Vehicle Surfaces". February 1977.
b. NATO Allied Environmental Engineering and Test Publication (AECTP) 400, Mechanical Environmental
Testing, Method 401, Vibration.
c. Beranek, Leo L., "Noise and Vibration Control" Revised Edition, 1988, Institute of Noise Control
Engineering, ISBN: 0-9622072-0-9.
d. Larkin, Paul A. and Smallwood, David O. "Control of an Acoustical Speaker System in a Reverberant
Chamber", Journal of the IEST, V. 47, No. 2, 2004.
e. Larkin, Paul A. and Hayes, Dann "Status of Direct Field Acoustic Testing", 27th Aerospace Testing
Seminar, October 16-18, 2012.
f. Maahs, Gordon, "Direct Field Acoustic Test (DFAT) Development and Flight Testing of Radiation Belt
Storm Probe (RBSP) Satellites", 27th Space Simulation Conference, November 5-8, 2012.
6.2 Related Documents.
a. NATO STANAG 4370, Environmental Testing.
b. NATO Allied Environmental Engineering and Test Publication (AECTP) 400, Mechanical Environmental
Testing, Method 402, Acoustic Noise.
c. ISO 266, Acoustics – Preferred Frequencies, International Organization for Standardization, 27 March 1997;
http://www.iso.org.
515.8-10

METHOD 515.8
d. Egbert, Herbert W. “The History and Rationale of MIL-STD-810 (Edition 2)”, January 2010; Institute of
Environmental Sciences and Technology, Arlington Place One, 2340 S. Arlington Heights Road, Suite 100,
Arlington Heights, IL 60005-4516.
e. IEST RP-DTE040.1, High-Intensity Acoustics Testing, Institute of Environmental Sciences and Technology,
Arlington Place One, 2340 S. Arlington Heights Road, Suite 100, Arlington Heights, IL 60005-4516;
.January 2003; http://www.iest.org.
f. NASA-STD-7001, Payload Vibroacoustic Test Criteria, National Aeronautics and Space Agency, 21 June
1996, http://standardsnasa.gov.
(Copies of Department of Defense Specifications, Standards, and Handbooks, and International
Standardization Agreements are available online at https://assist.dla.mil.
Requests for other defense-related technical publications may be directed to the Defense Technical Information Center
(DTIC), ATTN: DTIC-BR, Suite 0944, 8725 John J. Kingman Road, Fort Belvoir VA 22060-6218,
1-800-225-3842 (Assistance--selection 3, option 2), http://www.dtic.mil/dtic/; and the National Technical
Information Service (NTIS), Springfield VA 22161, 1-800-553-NTIS (6847), http://wwwntis.gov/.
515.8-11

METHOD 515.8
(This page is intentionally blank.)
515.8-12

METHOD 515.8 ANNEX A
3.2 Test Parameters.
For acoustic testing of external stores, the associated levels and definitions are shown in Table 515.8A-IV.
Table 515.8A-IV. Suggested acoustic test levels for assembled externally carried aircraft stores.
A = 6 dB/Octave when f > 400 Hz
0
A = 2 dB/Octave when f < 400 Hz
0
Functional Test
L = 20 log (q ) + 11 log (X) + 7 log (1-cos β) + G + H (dB) (see Notes 1, 5, 6, 7.)
0 1
f = 600 log (X/R) + C (see Notes 2, 3.)
0
Endurance Test
L = 20 log (q /q ) + 2.5 log (N/3T) + functional level (dB) (see Notes 1, 5, 6, 7.)
0 2 1
f = 600 log (X/R) + C (see Notes 2, 3.)
0
Definitions
q = captive flight dynamic pressure (lbs/ft2) < 1800
1
q = 1200 psf or maximum captive flight dynamic pressure (whichever is lower) (lbs/ft2)
2
N = maximum number of anticipated service missions (minimum N = 3)
R = local radius of store in inches (see Note 4.)
X = distance from nose of store along axis of store in inches
T = test time in hours (minimum T=1 hour unless otherwise specified)
C = -200 for locations (1) within one (D) of the aft end of the store, or (2) aft of a flow reentry point.
(See Note 8);
= 400 for all other locations
D = maximum store diameter in inches (see Note 4.)
β = local nose cone angle at X equals 1/tanβ = (R/X) (see figure 515.5A-2)
G = 72 unless measured data shows otherwise
E = 96 unless measured data shows otherwise
F = 84 unless measured data shows otherwise
H = 0 for 0.85 < M < 0.95;
=-3 dB for all other values of M
M = Mach number
515.8A-5

METHOD 515.8 ANNEX B
METHOD 515.8, ANNEX B
ADDITIONAL TECHNICAL GUIDANCE
1. REVERBERATION CHAMBERS.
A reverberation chamber is basically a cell with hard, acoustically reflective walls. When noise is generated in this
room, the multiple reflections within the main volume of the room cause a uniform diffuse noise field to be set up.
The uniformity of this field is disturbed by three main effects.
a. At low frequencies, standing modes are set up between parallel walls. The frequency below which these
modes become significant is related to the chamber dimensions. Small chambers, below about 100 cubic
meters in volume, are usually constructed so that no wall surfaces are parallel to each other in order to
minimize this effect.
b. Reflections from the walls produce higher levels at the surface. The uniform noise field therefore only applies
at positions within the central volume of the chamber; do not position test items within about 0.5 m (1.6 ft)
of the walls.
c. The size of the test item can distort the noise field if the item is large relative to the volume of the chamber.
It is normally recommended that the volume of the test item not exceed 10 percent of the chamber volume.
Noise is normally generated with an air modulator and is injected into the chamber via a coupling horn. Provision is
made in the chamber design to exhaust the air from the modulator through an acoustic attenuator in order to prevent
the direct transmission of high intensity noise to areas outside the test chamber.
2. PROGRESSIVE WAVE TUBES.
A parallel sided duct usually forms the working section of such a progressive noise facility. This may be circular or
rectangular in section to suit the test requirements. For testing panels, a rectangular section may be more suitable
while an aircraft carried store may be more conveniently tested in a duct of circular section. Noise is generated by an
air modulator coupled into one end of the working section by a suitable horn. From the opposite end of the plain duct
another horn couples the noise into an absorbing termination. Maximum absorption over the operating frequency
range is required here in order to minimize standing wave effects in the duct. Noise then progresses along the duct
and is applied with grazing incidence over the surface of the test item. The test item itself may be mounted within the
duct in which case the grazing incidence wave will be applied over the whole of its external surface. Alternatively,
the test item may be mounted in the wall of the duct when the noise will be applied to only that surface within the
duct, e.g., on one side of a panel. The method used will depend upon the test item and its in-service application.
3. ACOUSTIC NOISE CHARACTERISTICS.
Radiated high intensity noise is subjected to distortion due to adiabatic heating. Thus, due to heating of the high
pressure peaks and cooling of the rarefaction troughs, the local speed of propagation of these pressures is modified.
This causes the peaks to travel faster and the troughs to travel slower than the local speed of propagation such that, at
a distance from the source, a sinusoidal wave becomes triangular with a leading shock front. This waveform is rich
in harmonics and therefore the energy content is extended into a higher frequency range. It can be seen from this that
it is not possible to produce a pure sinusoidal tone at high noise intensities. The same effect takes place with high
intensity random noise that is commonly produced by modulating an airflow with a valve driven by a dynamic
actuator. Due to velocity and/or acceleration restraints on the actuator, it is not possible to modulate the airflow at
frequencies greater than about 1 kHz. Acoustic energy above this frequency, extending to 20 kHz or more, therefore
results from a combination of cold air jet noise and harmonic distortion from this lower frequency modulation.
4. CONTROL STRATEGIES.
Microphones are normally used to monitor and control the test condition. When testing stores and missiles, it is
recommended that not less than three microphones be used to control the test. Some test items may be more effectively
monitored on their vibration response; in which case, follow the monitoring requirements of Method 514.8, as
appropriate. Use a monitoring system capable of measuring random noise with a peak to rms ratio of up to 3.0.
Correct pressure calibrated microphones used in reverberation chambers for random incidence noise, while correcting
those used in progressive wave tubes for free field grazing incidence noise, and ensure both have a linear pressure
515.8B-1

METHOD 515.8 ANNEX B
response. Provide for averaging the outputs of the microphones to provide the spatial average of the noise for control
purposes.
5. DEFINITIONS.
5.1 Sound Pressure Level.
The sound pressure level (Lp) is the logarithmic ratio of the sound pressures:
l P
Lp=10log =20log
l P
0 0
Expressed as:
where l = reference intensity = 10-12 Wm-2
0
and P = reference pressure = 20 x 10-6 Pa
0
5.2 Third Octave Filters.
The center frequency, f , of a third octave filter is:
0
f = (f x f ) 1/2
0 1 2
where f = lower -3 dB frequency
1
and f = upper -3 dB frequency
2
The relationships between the upper and lower -3 dB frequencies are:
( f −f )
2 1 =0.23
f
0
1
f =2 3 f
2 1
Standard third octave bands are defined in International Specification ISO 266.
6. DIRECT FIELD ACOUSTIC NOISE CHARACTERISTICS.
Closed-loop, digital control is preferred for all direct field testing. Since the drivers used are capable of responding
over the entire test bandwidth (usually 25 to 10kHz) and beyond, narrow-band drive signals are often used to control
the test. Narrow-band control allows all features of modern, random vibration control systems such as; control of
local resonances, response limiting, peak limits/aborts based on spectral lines out, and rms limits/aborts to be used for
acoustic testing. See paragraph 6.1, references d and e, for more information about narrow-band control.
Single Input, Single Output (SISO) control will produce a well correlated sound field since the same drive signal is
delivered to all audio devices. However, sound pressure level (SPL) variations due to wave interference patterns in
the SISO field can be as large as + 12dB from the average SPL due to constructive and destructive wave combinations.
In addition, multiple microphone averaging can exacerbate the problem by allowing large variations at the control
points to result in an apparently well controlled composite when compared to the reference. Typical performance in
the SISO environment is + 1dB variation between the reference and the composite control average with + 5dB of
control microphone to control microphone variation and + 12dB or more between monitor microphone locations.
The recent application of MIMO (Multiple Input,-Multiple Output) acoustic control to the DFAN process has created
a much improved methodology over the earlier practice of using a SISO approach. The MIMO process is based on
multiple, independent inputs, multiple references and results in multiple independent drive outputs. Using MIMO
control the user can input magnitude, phase and coherence specifications with tolerance bands on each. The system
will use those constraints and the independent drives to produce a complying environment at each control point. In
effect, this method controls the response of each control microphone to meet its individual requirements based on the
input it receives from each independent drive signal. The result can be an incoherent field with minimum variation
between control microphones. This represents a huge improvement in field uniformity (spatial variation) as well as
providing a sound field with much lower coherence. Typical test tolerances for a MIMO controlled test are + 1dB on
515.8B-2

METHOD 515.8 ANNEX B
the composite control average and + 3dB at each control microphone relative to the reference spectrum for overall
levels in the 125 to 145dB range. See paragraph 6.1, reference e for more information about SISO and MIMO control.
As stated in paragraph 2.2.2.a.Ib, DFAN testing does not create an environment that is identical to a reverberant
chamber. However, DFAN testing can produce a very similar structural response. Paragraph 6.1, reference f gives a
detailed comparison of the results from a DFAN test with those from reverberant testing for a typical spacecraft
structure subjected to a typical launch vehicle environment. Similar configurations can be expected to produce similar
results. Structures and/or environments that vary greatly from those documented may require similar study and
evaluation before implementing the DFAN approach.
Lastly, a restriction on run time for test levels above 144dB is usually imposed due to heat build-up in the driving
coils of the acoustic drivers. Currently technology limits runs at 140 to 144dB to about one minute and runs above
144dB to about 30 seconds. If run times longer than these are required, it is recommended that the total test time be
broken into multiple segments of 30 seconds each until the total run time is accumulated.
515.8B-3

METHOD 515.8 ANNEX B
(This page is intentionally blank.)
515.8B-4

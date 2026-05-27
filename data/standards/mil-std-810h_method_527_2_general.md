---
standard: MIL-STD-810H
method: "527.2"
category: general
language: en
---

# MIL-STD-810H Method 527.2 — MULTI-EXCITER TEST

METHOD 527.2
METHOD 527.2
MULTI-EXCITER TEST
CONTENTS
Paragraph Page
1. SCOPE ........................................................................................................................................................... 1
1.1 PURPOSE………………………………………………………………. .......................................................... 1
1.2 APPLICATION ................................................................................................................................................... 1
1.2.1 GENERAL DISCUSSION ..................................................................................................................................... 1
1.2.2 TERMINOLOGY ................................................................................................................................................. 2
1.3 LIMITATIONS .................................................................................................................................................... 5
2. TAILORING GUIDANCE ........................................................................................................................... 5
2.1 SELECTING THE MET METHOD ........................................................................................................................ 5
2.1.1 EFFECTS OF THE MET ENVIRONMENT ............................................................................................................. 5
2.1.2 SEQUENCE AMONG OTHER METHODS.............................................................................................................. 6
2.2 SELECTING A PROCEDURE ................................................................................................................................ 6
2.2.1 PROCEDURE SELECTION CONSIDERATIONS ...................................................................................................... 6
2.3 DETERMINE TEST LEVELS AND CONDITIONS ................................................................................................... 6
2.3.1 LABORATORY TEST DATA INPUT ..................................................................................................................... 7
2.3.1.1 CROSS-SPECTRAL DENSITY CONSIDERATIONS ................................................................................................. 7
2.3.1.2 GENERAL ......................................................................................................................................................... 7
2.3.2 LABORATORY TEST OUTPUT ............................................................................................................................ 7
2.4 TEST ITEM OPERATION .................................................................................................................................... 7
3. INFORMATION REQUIRED ..................................................................................................................... 7
3.1 PRETEST ........................................................................................................................................................... 7
3.2 DURING TEST ................................................................................................................................................... 8
3.3 POST-TEST ....................................................................................................................................................... 8
4. TEST PROCESS ........................................................................................................................................... 8
4.1 TEST FACILITY ................................................................................................................................................. 8
4.2 CONTROLS ....................................................................................................................................................... 8
4.2.1 CALIBRATION ................................................................................................................................................... 9
4.2.2 TOLERANCES.................................................................................................................................................... 9
4.3 TEST INTERRUPTION ........................................................................................................................................ 9
4.3.1 INTERRUPTION DUE TO LABORATORY EQUIPMENT MALFUNCTION ................................................................. 9
4.3.2 INTERRUPTION DUE TO TEST ITEM OPERATION FAILURE ................................................................................ 9
4.3.3 INTERRUPTION DUE TO A SCHEDULED EVENT .............................................................................................. 10
4.3.4 INTERRUPTION DUE TO EXCEEDING TEST TOLERANCES ................................................................................ 10
4.4 TEST SETUP .................................................................................................................................................... 10
4.4.1 INSTRUMENTATION ........................................................................................................................................ 10
4.4.2 PLATFORM INTEGRATION ............................................................................................................................... 11
4.4.3 SETUP ANALYSIS ........................................................................................................................................... 11
4.5 TEST EXECUTION ........................................................................................................................................... 11
4.5.1 PREPARATION FOR TEST................................................................................................................................. 11
4.5.1.1 PRELIMINARY STEPS ...................................................................................................................................... 12
4.5.1.2 PRETEST STANDARD AMBIENT CHECKOUT .................................................................................................... 12
4.5.2 PROCEDURE ................................................................................................................................................... 12
527.2-i

METHOD 527.2
CONTENTS-Continued
Paragraph Page
5. ANALYSIS OF RESULTS ......................................................................................................................... 13
5.1 PHYSICS OF FAILURE ...................................................................................................................................... 14
5.2 QUALIFICATION TESTS ................................................................................................................................... 14
5.3 OTHER TESTS ................................................................................................................................................. 14
6. REFERENCE/RELATED DOCUMENTS ............................................................................................... 14
6.1 REFERENCED DOCUMENTS ............................................................................................................................. 14
6.2 RELATED DOCUMENTS ................................................................................................................................... 15
FIGURES
FIGURE 527.2-1. SESA - SINGLE EXCITER VERTICAL AXIS TEST SETUP ........................................................................ 3
FIGURE 527.2-2. MESA (IF CONTROL CONFIGURED FOR TWO EXCITER 1-DOF MOTION) OR MEMA (IF CONTROL AND
MECHANICAL COUPLINGS CONFIGURED FOR TWO EXCITER 2-DOF MOTION) ..................................... 3
FIGURE 527.2-3. MEMA - TRI-AXIAL EXCITER TEST SETUP (TRANSLATIONAL DEGREES-OF-FREEDOM) ..................... 4
METHOD 527.2 ANNEX A
ENGINEERING INFORMATION FOR MET TRANSDUCER PLACEMENT
1. GENERAL PHILOSOPHY FOR A MET .............................................................................................. A-1
2. REFERENCE POINT CONSIDERATIONS FOR MDOF TESTING ................................................ A-1
2.1 REFERENCE DATA CONSIDERATIONS ........................................................................................................... A-1
2.2 REFERENCE POINT KENMATICS.................................................................................................................... A-1
ANNEX A FIGURE
FIGURE 527.2A-1. BODY WITH n ACCELEROMETERS. PLACEMENTS........................................................................ A-2
METHOD 527.2 ANNEX B
SYSTEM IDENTIFICATION FOR LINEAR TIME-INVARIANT MDOF SYSTEMS
1. TRANSFER-FUNCTION ESTIMATIONS ............................................................................................ B-1
2. SIGNAL TRANSFORMATION .............................................................................................................. B-1
3. CONTROL IMPLEMENTATION .......................................................................................................... B-1
3.1 SISO AUTO AND CROSS-SPECTRAL DEFINITIONS REVIEW .......................................................................... B-1
3.2 SISO TRANSFER FUNCTION AND COHERENCE FUNCTION DEFINITIONS REVIEW ......................................... B-2
3.3 MIMO AUTO-SPECTRA, CROSS-SPECTRA, AND INITIAL FUNCTION ESTIMATES .......................................... B-3
3.3.1 FREQUENCY DOMAIN TRANSFER FUNCTION RELATIONSHIP ........................................................................ B-3
3.3.2 KEY TRANSFER FUNCTION DERIVATIONS ................................................................................................... B-4
3.3.3 KEY TRANSFER FUNCTION DERIVATIONS ALTERNATIVE ............................................................................. B-5
3.4 MIMO COHERENCE FUNCTIONS .................................................................................................................. B-6
3.4.1 ORDINARY COHERENCE ............................................................................................................................... B-6
3.4.2 PARTIAL COHERENCE .................................................................................................................................. B-7
3.4.3 MULTIPLE COHERENCE ................................................................................................................................ B-7
3.5 DRIVE SIGNAL COMPENSATION ................................................................................................................... B-7
527.2-ii

METHOD 527.2
CONTENTS-Continued
Paragraph Page
METHOD 527.2 ANNEX C
PROCEDURE I MET (TIME WAVEFORM REPLICATION (TWR) SPECIFIC)
1. PROCEDURE I MET (TIME DOMAIN REFERENCE CRITERIA) ................................................ C-1
1.1 PREPROCESSING ........................................................................................................................................... C-1
2. ANALYSIS CONSIDERATIONS FOR A PROCEDURE I MET ....................................................... C-1
2.1 ADDRESSING TRANSLATIONAL MOTION ...................................................................................................... C-1
2.2 ADDRESSING ANGULAR MOTION ................................................................................................................. C-1
3. TEST TOLERANCES FOR A PROCEDURE I MET .......................................................................... C-2
3.1 COMPOSITE (GLOBAL) ERROR DISCUSSION FOR PROCEDURE I .................................................................... C-2
3.2 GLOBAL RMS ERROR .................................................................................................................................. C-2
3.3 GLOBAL ASD ERROR ................................................................................................................................... C-4
3.4 GLOBAL SRS ERROR.................................................................................................................................... C-6
METHOD 527.2 ANNEX D
PROCEDURE II MET (SPECTRAL DENSITY MATRIX (SDM) SPECIFIC)
1. PROCEDURE II MET (FREQUENCY DOMAIN REFERENCE CRITERIA) ................................ D-1
1.1 PREPROCESSING ........................................................................................................................................... D-1
2. ANALYSIS CONSIDERATIONS FOR A PROCEDURE II MET ...................................................... D-1
2.1 MESA AND MEMA SPECIFICATION PARAMETERS ...................................................................................... D-1
2.1.1 CROSS SPECTRAL DENSITY STRUCTURE ...................................................................................................... D-2
2.2 CONTROL HIERARCHY ................................................................................................................................. D-2
2.2.1 MEASURED DATA AVAILABLE ..................................................................................................................... D-2
2.2.2 MEASURED DATA NOT AVAILABLE ............................................................................................................. D-2
2.2.3 USE OF 1-DOF REFERENCES ........................................................................................................................ D-3
3. TEST TOLERANCES FOR A PROCEDURE II MET ......................................................................... D-3
3.1 COMPOSITE (GLOBAL) ERROR DISCUSSION FOR PROCEDURE II ................................................................... D-3
ANNEX D TABLES
TABLE 527.2D-I. REFERENCE CRITERIA FOR A 2-DOF LINEAR MOTION RANDOM MET ......................................... D-1
TABLE 527.2D-II. REFERENCE CRITERIA FOR A 3-DOF LINEAR MOTION RANDOM MET ......................................... D-1
METHOD 527.2 ANNEX E
LABORATORY VIBRATION TEST SCHEDULE DEVELOPMENT
FOR MULTI-EXCITER APPLICATIONS
1. SCOPE ....................................................................................................................................................... E-1
2. FACILITIES AND INSTRUMENTATION ........................................................................................... E-1
2.1 FACILITIES ................................................................................................................................................... E-1
527.2-iii

METHOD 527.2
CONTENTS-Continued
Paragraph Page
2.2 INSTRUMENTATION ...................................................................................................................................... E-1
3. REQUIRED TEST CONDITIONS .......................................................................................................... E-1
3.1 TEST CONFIGURATIONS ............................................................................................................................... E-1
3.1.1 BASIC REPRESENTATION OF A MIMO SYSTEM ............................................................................................ E-2
3.1.2 GENERALIZED REPRESENTATION OF A MIMO SYSTEM ............................................................................... E-2
3.2 GENERALIZED MDOF VIBRATION CONTROL DISCUSSION ........................................................................... E-3
4. TEST PROCEDURES .............................................................................................................................. E-4
4.1 DEVELOPMENT OF MISSION OR LIFETIME SCENARIO ................................................................................... E-4
4.2 LIMITATIONS ................................................................................................................................................ E-4
4.3 FIELD DATA ACQUISITION ........................................................................................................................... E-5
4.3.1 INSTRUMENTATION ...................................................................................................................................... E-5
4.4 USE OF RIGID BODY MODES ........................................................................................................................ E-5
4.4.1 ACCELERATION (INPUT) TRANSFORMATION ................................................................................................ E-5
4.4.1.1 ACCELERATION (INPUT) TRANSFORMATION DERIVATION ........................................................................... E-6
4.4.1.2 EQUATION 4.1 .............................................................................................................................................. E-6
4.4.2 DRIVE (OUTPUT) TRANSFORMATION ........................................................................................................... E-7
4.4.2.1 DRIVE (OUTPUT) TRANSFORMATION DERIVATION ...................................................................................... E-8
4.5 DATA ANALYSIS .......................................................................................................................................... E-9
4.5.1 PHASE AND COHERENCE BASED REPRESENTATIONS OF CSD TERMS ........................................................ E-10
4.5.2 POSITIVE DEFINITE SDM CONSIDERATIONS .............................................................................................. E-10
4.5.3 DATA COMPRESSION .................................................................................................................................. E-11
4.5.4 LIMITING STRATEGIES ............................................................................................................................... E-12
4.5.5 MINIMUM DRIVE CONSIDERATIONS ........................................................................................................... E-12
4.5.5.1 INDEPENDENT DRIVES................................................................................................................................ E-12
4.6 INDEPENDENT REFERENCES ....................................................................................................................... E-13
4.7 RECOMMENDED PRACTICES SUMMARY ..................................................................................................... E-14
5. DATA REQUIRED ................................................................................................................................. E-15
5.1 REFERENCE SDM DEVELOPMENT .............................................................................................................. E-15
5.1.1 SDM ENSEMBLE CSD CHARACTERISTICS ................................................................................................. E-15
5.2 TEST TOLERANCE RECOMMENDATIONS ..................................................................................................... E-16
5.3 LABORATORY DATA .................................................................................................................................. E-16
6. MDOF VSD METHODS ........................................................................................................................ E-16
6.1 OPTIONS CONSIDERED ............................................................................................................................... E-16
6.1.1 METHOD I .................................................................................................................................................. E-16
6.1.2 METHOD II ................................................................................................................................................. E-18
6.2 EXAMPLE ................................................................................................................................................... E-19
6.3 CONCLUDING REMARKS ............................................................................................................................ E-25
APPENDIX A GLOSSARY ............................................................................................................................... E-27
APPENDIX B ABBREVIATIONS .................................................................................................................... E-29
APPENDIX C NOMENCLATURE.................................................................................................................... E-31
APPENDIX D MATRIX ALGEBRA REVIEW ................................................................................................. E-33
APPENDIX E REFERENCES ........................................................................................................................... E-37
527.2-iv

METHOD 527.2
METHOD 527.2
MULTI-EXCITER TEST
NOTE: Tailoring is required. Select methods, procedures, and parameter levels based on the
tailoring process described in Part One, paragraph 4, and Annex C. Apply the general guidelines for
laboratory test methods described in Part One, paragraph 5 of this standard.
Although various forms of multi-exciter test (MET) have been discussed in the technical literature and conducted in
the laboratory dating back over multiple decades, there are still many issues regarding standardization of laboratory
MET. In this early version of the Multi-Exciter Test Method, the intent is to introduce the basic definitions and
structure of a laboratory-based multi-exciter test. MET hardware and control algorithms have continued to improve
at an impressive rate recently, and MET is becoming more common in many dynamic test facilities. Feedback from
the growing MET user community is highly encouraged, will be reviewed, and will play a major role in improving
this Method.
Organization. The main body of this Method is arranged similarly to that of other methods of MIL-STD-810. A
considerable body of supplementary information is included in the Annexes. Reference citations to external
documents are at the end of the main body (paragraph 6.1). The Annexes are structured as follows:
ANNEX A - ENGINEERING INFORMATION FOR MET TRANSDUCER PLACEMENT
ANNEX B - SYSTEM IDENTIFICATION FOR LINEAR TIME INVARIANT MDOF SYSTEMS
ANNEX C - PROCEDURE I MET (TIME WAVEFORM REPLICATION (TWR) SPECIFIC)
ANNEX D - PROCEDURE II MET (SPECTRAL DENSITY MATRIX (SDM) SPECIFIC)
ANNEX E - LABORATORY VIBRATION TEST SCHEDULE DEVELOPMENT FOR
MULTI-EXCITER APPLICATIONS
1. SCOPE.
1.1 Purpose.
Multi-exciter test methodology is performed to demonstrate, or provide a degree of confidence if multiple test items
are considered, that materiel can structurally and functionally withstand a specified dynamic environment, e.g.,
stationary, non-stationary, or of a shock nature, that must be replicated on the test item in the laboratory with more
than one motion degree-of-freedom. The laboratory test environment may be derived from field measurements on
materiel, or may be based on an analytically-generated specification.
1.2 Application.
a. General. Use this Method for all types of materiel except as noted in Part One, paragraph 1.3, and as stated
in paragraph 1.3 below. For combined environment tests, conduct the test in accordance with the
applicable test documentation. However, use this Method for determination of dynamic test levels,
durations, data reduction, and test procedure details.
b. Purpose of Test. The test procedures and guidance herein are adaptable to various test purposes including
development, reliability, qualification, etc.
c. Dynamics Life Cycle. Table 514.8-I provides an overview of various life cycle situations during which
some form of vibration (stationary or non-stationary) may be encountered, along with the anticipated
platform involved.
1.2.1 General Discussion.
Use this Method to demonstrate that the materiel of interest can structurally and functionally withstand a specified
dynamic environment that is defined in more than a single-degree-of-freedom (SDOF) motion; i.e., in multiple-degree-
527.2-1

METHOD 527.2
of-freedom (MDOF) motion. Establishing confidence intervals may also be of interest if multiple like items are under
test. Specification of the environment may be through a detailed summary of measured field data related to the test
materiel that entails more than one degree-of-freedom, or analytical generation of an environment that has been
properly characterized in MDOF. In general, specification of the environment will include several degrees of freedom
in a materiel measurement point configuration, and testing of the materiel in the laboratory in a SDOF mode is
considered inadequate to properly distribute vibration energy in the materiel in order to satisfy the specification. As
a result of the increased complexity of application of MET over multiple application of SDOF single-exciter testing
(SET), an analyst, after careful review of the available data and specification, will need to provide rationale for
selection of this Method. Methods 514.8, 516.8, 519.8, and 525.2 provide guidance in developing the rationale and
requirement for MET.
Reasons for selection of MET over SET may include the following.
a. MET provides a distribution of vibration or shock energy to the materiel in more than one axis in a
controlled manner without relying on the dynamics of the materiel for such distribution.
b. MET may be selected when the physical configuration of the materiel is such that its slenderness ratio is
high, and SET must rely on the dynamics of the materiel to distribute energy.
c. For large and heavy test materiel, more than one exciter may be necessary to provide sufficient energy to
the test item.
d. MET allows more degrees-of-freedom in accounting for both the impedance matches and the in service
boundary conditions of the materiel.
1.2.2 Terminology.
Several terms need to be carefully defined for contrasting MET with SET. The term “test configuration” used in this
document will refer to the totality of description for laboratory testing including the sources of excitation, test item
fixturing, and orientation. In either testing configuration, distinction must be made between excitation measurement
in a vector axis of excitation, and measurement on the test item in either the vector axis of excitation or in another
vector different from the vector axis of excitation. Generally, to avoid confusion in specification and reporting, the
vector directions of excitation and measurement must be specified in terms of a single laboratory inertial frame of
reference related to the test configuration. In addition, it is helpful to specify the test item geometrical configuration
along with the dynamic properties such as mass moments of inertia relative to the single laboratory inertial frame of
reference.
a. Single-Degree-of-Freedom (SDOF) – motion defined by materiel movement along or about a single axis
whose description requires only one coordinate to completely define the position of the item at any instant.
b. Multi-Degree-of-Freedom (MDOF) – motion defined by test item movement along or about more than one
axis whose description requires two or more coordinates to completely define the position of the item at any
instant.
c. Single-Axis (SA) - excitation or response measurement in a unique single vector direction (linear or
rotational). For rotational axis, the vector direction is perpendicular to the plane of rotation of the exciter or
test item. Figure 527.2-1 displays a single-axis input in the vertical direction to an extended structure.
d. Multi-Axis (MA) – excitation or response measurement that requires more than one unique vector for
description. Refer to Figures 527.2-2 and 527.2-3 for MA examples of both two-axis and three-axis inputs
to a common structure.
e. Single-Exciter/Single-Axis (SESA) - application of a single exciter providing dynamic input to the test item
in a single vector direction. All SET configurations are SESA by definition.
f. Multi-Exciter/Single-Axis (MESA) – application of multiple exciters providing dynamic input to the test
item in a single vector direction. For example, extended materiel might require excitation at the forward and
aft end in a single vector axis as illustrated in Figure 527.2-2. If the definition of excitation requires more
than a single vector, refer to the MEMA definition.
527.2-2

METHOD 527.2
1.3 Limitations.
This Method addresses very general testing configurations for applying excitation in multiple axes to materiel.
Generally, field deployed materiel has boundary (or impedance) conditions that are very difficult and often cost
prohibitive to replicate in laboratory testing. The overall goal of a MET is to achieve a distribution of materiel
excitation energy that approaches that appearing during in-service deployment, while minimizing the difference
between in-service and laboratory boundary conditions. Fixturing design limitations and/or other physical constraints
may limit application of in-service environment in the laboratory. Also, in-service measurements may not be adequate
to specify the laboratory test configuration. As always, engineering analysis and judgment will be required to ensure
the test fidelity is sufficient to meet the test objectives.
The following limitations also apply to this Method:
a. It does not address aspects of vendor-supplied software control strategy for a MET.
b. It does not address advantages or disadvantages of Procedure I and Procedure II MET as defined in paragraph
2.2. The state of the art in a MET is not such that a comprehensive comparison can be made at this time.
c. It does not address optimization techniques of the laboratory test configuration relative to distribution of the
excitation energy within the test item.
d. It does not address technical issues related to axes of excitation and materiel mass and product moments of
inertia. Nor does it address the need for specialized software for optimizing the axes of excitation with
respect to mass and products of inertia.
e. It generally does not provide specific test tolerance information that is highly dependent on the (1) test
objective, (2) test laboratory measurement configuration, and (3) vendor software control strategy.
f. It does not discuss, in detail, the potential for efficiencies and efficacies of a MET over SET, leaving this as
a part of specification of MET peculiar to the in-service measured environment.
g. It does not discuss optimum in-service measurement configuration factors consistent with a MET.
h. It assumes that excitation is provided mechanically through electro-dynamic or servo-hydraulic exciters, and
does not consider combined acoustic (refer to Method 523.4) or pneumatic induced modes of excitation.
2. TAILORING GUIDANCE.
2.1 Selecting the MET Method.
After examining requirements documents and applying the tailoring process in Part One of this Standard to determine
where significant excitation energy distribution effects are foreseen in the life cycle of the materiel, or substantial
testing cost savings might be achieved by employing MET strategy, use the following to confirm the need for this
Method, and to place it in sequence with other Methods.
2.1.1 Effects of the MET Environment.
In general, all in-service measured environments require multiple axis response measurements for complete
description. Generally, a MET will distribute excitation energy to the test item and minimize the effects of in-service
boundary conditions. The following is a partial list of effects to materiel that may be better replicated in the laboratory
under a MET than a SET.
a. Fatigue, cracking, and rupture sensitive to multi-axis excitation.
b. Deformation of materiel structure, e.g., protruding parts.
c. Loosening of seals and connections.
d. Displacement of components.
e. Chafing of surfaces with single-axis design.
f. Contact, short-circuiting, or degradation of electrical components.
g. Misalignment of materiel components (e.g., optical).
527.2-5

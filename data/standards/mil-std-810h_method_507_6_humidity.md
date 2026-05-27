---
standard: MIL-STD-810H
method: "507.6"
category: humidity
language: en
---

# MIL-STD-810H Method 507.6 — HUMIDITY

METHOD 507.6
METHOD 507.6
HUMIDITY
CONTENTS
1. SCOPE ........................................................................................................................................................... 1
1.1 PURPOSE .......................................................................................................................................................... 1
1.2 APPLICATION ................................................................................................................................................... 1
1.3 LIMITATIONS .................................................................................................................................................... 1
2. TAILORING GUIDANCE ........................................................................................................................... 1
2.1 SELECTING THE HUMIDITY METHOD ............................................................................................................... 1
2.1.1 EFFECTS OF WARM, HUMID ENVIRONMENTS ................................................................................................... 2
2.1.2 SEQUENCE AMONG OTHER METHODS.............................................................................................................. 2
2.2 SELECTING PROCEDURES ................................................................................................................................. 2
2.2.1 PROCEDURE SELECTION CONSIDERATIONS ...................................................................................................... 3
2.2.2 DIFFERENCE BETWEEN PROCEDURES .............................................................................................................. 3
2.3 DETERMINE TEST LEVELS, CONDITIONS, AND DURATIONS .............................................................................. 3
2.3.1 TEST TEMPERATURE - HUMIDITY ..................................................................................................................... 3
2.3.2 TEST DURATION ............................................................................................................................................... 5
2.3.3 TEST ITEM CONFIGURATION ............................................................................................................................ 7
2.3.4 ADDITIONAL GUIDELINES ................................................................................................................................ 7
2.4 OPERATIONAL CHECKOUT ............................................................................................................................... 7
2.4.1 PROCEDURE I (INDUCED CYCLES B1, B2, OR B3, FOLLOWED BY NATURAL CYCLES B1, B2, OR B3) ............. 7
2.4.2 PROCEDURE II - AGGRAVATED ........................................................................................................................ 7
2.5 TEST VARIATIONS ............................................................................................................................................ 7
2.6 PHILOSOPHY OF TESTING ................................................................................................................................. 7
2.6.1 PROCEDURE I - INDUCED (STORAGE AND TRANSIT) CYCLES ........................................................................... 8
2.6.2 PROCEDURE I - NATURAL CYCLES ................................................................................................................. 10
2.6.3 PROCEDURE II - AGGRAVATED CYCLE (FIGURE 507.6-7) .............................................................................. 13
3. INFORMATION REQUIRED ................................................................................................................... 14
3.1 PRETEST ......................................................................................................................................................... 14
3.2 DURING TEST ................................................................................................................................................. 15
3.3 POST-TEST ..................................................................................................................................................... 15
4. TEST PROCESS ......................................................................................................................................... 15
4.1 TEST FACILITY ............................................................................................................................................... 15
4.1.1 GENERAL DESCRIPTION ................................................................................................................................. 15
4.1.2 FACILITY DESIGN ........................................................................................................................................... 15
4.1.3 TEST SENSORS AND MEASUREMENTS ............................................................................................................ 15
4.1.4 AIR VELOCITY ............................................................................................................................................... 16
4.1.5 HUMIDITY GENERATION ................................................................................................................................ 16
4.1.6 CONTAMINATION PREVENTION ...................................................................................................................... 16
4.2 CONTROLS ..................................................................................................................................................... 16
4.3 TEST INTERRUPTION ...................................................................................................................................... 16
4.3.1 INTERRUPTION DUE TO CHAMBER MALFUNCTION ......................................................................................... 16
4.3.2 INTERRUPTION DUE TO TEST ITEM OPERATION FAILURE ............................................................................... 17
4.4 TEST EXECUTION ........................................................................................................................................... 17
4.4.1 PREPARATION FOR TEST................................................................................................................................. 17
4.4.1.1 TEST SETUP .................................................................................................................................................... 17
4.4.1.2 PRELIMINARY STEPS ...................................................................................................................................... 17
507.6-i

METHOD 507.6
CONTENTS - Continued
Paragraph Page
4.4.1.3 PRETEST CHECKOUT ...................................................................................................................................... 17
4.4.2 TEST PROCEDURES ......................................................................................................................................... 17
4.4.2.1 PROCEDURE I - STORAGE AND TRANSIT CYCLES (CYCLES B2 OR B3), AND NATURAL (CYCLES B1,
B2, OR B3 ...................................................................................................................................................... 17
4.4.2.2 PROCEDURE II - AGGRAVATED ...................................................................................................................... 18
5. ANALYSIS OF RESULTS ......................................................................................................................... 19
6. REFERENCE/RELATED DOCUMENTS ............................................................................................... 19
6.1 REFERENCED DOCUMENTS ............................................................................................................................. 19
6.2 RELATED DOCUMENTS ................................................................................................................................... 19
TABLES
TABLE 507.6-I HIGH HUMIDITY DIURNAL CATEGORIES .......................................................................................... 5
TABLE 507.6-II TEST CYCLES (DAYS) ...................................................................................................................... 7
TABLE 507.6-III CONSTANT TEMPERATURE AND HUMIDITY - INDUCED CYCLE B1 ................................................... 8
TABLE 507.6-IV CYCLIC HIGH RELATIVE HUMIDITY - INDUCED CYCLE B2 .............................................................. 9
TABLE 507.6-V HOT HUMID - INDUCED CYCLE B3 ................................................................................................. 10
TABLE 507.6-VI CONSTANT TEMPERATURE AND HUMIDITY - NATURAL CYCLE B1 ................................................ 11
TABLE 507.6-VII CYCLIC HIGH RELATIVE HUMIDITY - NATURAL CYCLE B2 ........................................................... 12
TABLE 507.6-VIII HOT HUMID - NATURAL CYCLE B3................................................................................................ 13
TABLE 507.6-IX AGGRAVATED CYCLE .................................................................................................................... 14
FIGURES
FIGURE 507.6-1 INDUCED CYCLE B1 - STORAGE AND TRANSIT................................................................................. 8
FIGURE 507.6-2 INDUCED CYCLE B2 - STORAGE AND TRANSIT................................................................................. 9
FIGURE 507.6-3 INDUCED CYCLE B3 - STORAGE AND TRANSIT............................................................................... 10
FIGURE 507.6-4 NATURAL CYCLE B1 - CONSTANT HIGH HUMIDITY ....................................................................... 11
FIGURE 507.6-5 NATURAL CYCLE B2 - CYCLIC HIGH HUMIDITY ............................................................................ 12
FIGURE 507.6-6 NATURAL CYCLE B3 - HOT HUMID................................................................................................ 13
FIGURE 507.6-7 AGGRAVATED TEMPERATURE - HUMIDITY CYCLE ........................................................................ 14
METHOD 507.6, ANNEX A
PHYSICAL PHENOMENA ASSOCIATED WITH HUMIDITY
1. ABSORPTION .......................................................................................................................................... A-1
2. ADSORPTION .......................................................................................................................................... A-1
3. BREATHING ............................................................................................................................................ A-1
4. CONDENSATION .................................................................................................................................... A-1
5. DIFFUSION ............................................................................................................................................... A-1
507.6-ii

METHOD 507.6
2.1.1 Effects of Warm, Humid Environments.
Temperature-humidity conditions have physical and chemical effects on materiel; the temperature and humidity
variations can also trigger synergistic effects or condensation inside materiel. Consider the following typical
problems to help determine if this Method is appropriate for the materiel being tested. This list is not intended to be
all-inclusive.
a. Surface effects, such as:
(1) Oxidation and/or galvanic corrosion of metals.
(2) Increased chemical reactions.
(3) Chemical or electrochemical breakdown of organic and inorganic surface coatings.
(4) Interaction of surface moisture with deposits from external sources to produce a corrosive film.
(5) Changes in friction coefficients, resulting in binding or sticking.
b. Changes in material properties, such as:
(1) Swelling of materials due to sorption effects.
(2) Other changes in properties.
(a) Loss of physical strength.
(b) Electrical and thermal insulating characteristics.
(c) De-lamination of composite materials.
(d) Change in elasticity or plasticity.
(e) Degradation of hygroscopic materials.
(f) Degradation of explosives and propellants by absorption.
(g) Degradation of optical element image transmission quality.
(h) Degradation of lubricants.
c. Condensation and free water, such as:
(1) Electrical short circuits.
(2) Fogging of optical surfaces.
(3) Changes in thermal transfer characteristics.
2.1.2 Sequence Among Other Methods.
a. General. Use the anticipated life cycle sequence of events as a general sequence guide (see Part One,
paragraph 5.5).
b. Unique to this Method. Humidity testing may produce irreversible effects. If these effects could
unrealistically influence the results of subsequent tests on the same item(s), perform humidity testing
following those tests. Also, because of the potentially unrepresentative combination of environmental
effects, it is generally inappropriate to conduct this test on the same test sample that has previously been
subjected to salt fog, sand and dust, or fungus tests. Dynamic environments (vibration and shock) could
influence the results of humidity testing. Consider performing these dynamic tests prior to humidity tests.
2.2 Selecting Procedures
This Method consists of two procedures, Procedure I (Induced (Storage and Transit) and Natural Cycles), and
Procedure II (Aggravated). Determine the procedure(s) to be used.
NOTE: The materiel’s anticipated Life Cycle Environmental Profile (LCEP) may reveal other
scenarios that are not specifically addressed in the procedures. Tailor the procedures as necessary
to capture the LCEP variations, but do not reduce the basic test requirements reflected in the below
procedures. (See paragraph 2.3 below.)
507.6-2

METHOD 507.6
2.2.1 Procedure Selection Considerations.
a. The operational purpose of the materiel.
b. The natural exposure circumstances.
c. Test data required to determine if the operational purpose of the materiel has been met.
d. Test duration.
2.2.2 Difference Between Procedures. (See paragraph 1.3c for related information on limitations.)
a. Procedure I – Induced (Storage and Transit) and Natural Cycles. Once a cycle is selected, perform the storage
and transit portion first, followed by the corresponding natural environment portion of the cycle. Procedure I
includes:
(1) three unique cycles that represent conditions that may occur during storage or transit, as well as
(2) three unique natural environment cycles that are performed on test items that are open to the environment.
NOTE: Although combined under one major column in Table 507.6-I, storage
configurations (and any packaging) may differ from configurations for the transit mode (see
paragraph 2.3.3). Ensure the configuration used for testing is appropriate for the intended
portion of the LCEP. Items in storage or transit could also experience relatively constant
conditions if situated near heat-producing equipment, or are sufficiently insulated from
external cycling conditions. For the purpose of this test, a “sealed” item is one that could
have a relatively high internal level of humidity and lacks continuous or frequent ventilation.
It does not include hermetically sealed items.
The internal humidity may be caused by these or other mechanisms:
(a) Entrapped, highly humid air.
(b) Presence of free water.
(c) Penetration of moisture through test item seals (breathing).
(d) Release of water or water vapor from hygroscopic material within the test item.
b. Procedure II – Aggravated. Procedure II exposes the test item to more extreme temperature and humidity
levels than those found in nature (without contributing degrading elements), but for shorter durations. Its
advantage is that it produces results quickly, i.e., it may, generally, exhibit temperature-humidity effects
sooner than in the natural or induced procedures. Its disadvantage is that the effects may not accurately
represent those that will be encountered in actual service. Be careful when interpreting results. This
procedure is used to identify potential problem areas, and the test levels are fixed.
2.3 Determine Test Levels, Conditions, and Durations.
Related test conditions depend on the climate, duration, and test item configuration during shipping, storage, and
deployment. The variables common to both procedures are the temperature-humidity cycles, duration, and
configuration. These variables are discussed below. Requirements documents may impose or imply additional test
conditions. Otherwise, use the worst-case conditions to form the basis for selecting the test and test conditions to use.
2.3.1 Test Temperature - Humidity.
The specific test temperature - humidity values are selected, preferably, from the requirements documents. If this
information is not available, base determination of the test temperature-humidity values for Procedure I on the world
geographical areas in which the test item will be used, plus any additional considerations. Table 507.6-I was
developed from AR 70-38 (paragraph 6.1, reference a), MIL-HDBK-310 (paragraph 6.1, reference b), NATO
STANAG 4370 (paragraph 6.1, reference d), AECTP 200 (paragraph 6.1, reference e), and NATO STANAG 4370,
AECTP 230 (paragraph 6.1, reference c, (part three)) and includes the temperature and relative humidity conditions
for three geographical categories where high relative humidity conditions may be of concern, and three related
categories of induced conditions. The temperature and humidity data are those used in the source documents
507.6-3

METHOD 507.6
mentioned above. The cycles were derived from available data; other geographic areas could be more severe. For
Procedure I, the temperature and humidity levels in Table 507.6-I are representative of specific climatic areas; the
natural cycles are not adjustable. Figures 507.6-1 through 507.6-6 are visual representations of the cycles in Table
507.6-I.
Although they occur briefly or seasonally in the mid-latitudes, basic high humidity conditions are found most often
in tropical areas. One of the two high humidity cycles (constant high humidity) represents conditions in the heavily
forested areas where nearly constant conditions may prevail during rainy and wet seasons. Exposed materiel is
likely to be constantly wet or damp for many days at a time. A description of each category follows.
a. Constant high humidity (Cycle B1). Constant high humidity is found most often in tropical areas, although it
occurs briefly or seasonally in the mid-latitudes. The constant-high-humidity cycle represents conditions in
heavily forested areas where nearly constant temperature and humidity may prevail during rainy and wet
seasons with little (if any) solar radiation exposure. Tropical exposure in a tactical configuration or mode is
likely to occur under a jungle canopy. Exposed materiel is likely to be constantly wet or damp for many days
at a time. World areas where these conditions occur are the Congo and Amazon Basins, the jungles of Central
America, Southeast Asia (including the East Indies), the north and east coasts of Australia, the east coast of
Madagascar, and the Caribbean Islands. The conditions can exist for 25 to 30 days each month in the most
humid areas of the tropics. The most significant variation of this cycle is its frequency of occurrence. In
many equatorial areas, it occurs monthly, year round, although many equatorial areas experience a distinctive
dry season. The frequency decreases as the distance from the equator increases. The mid-latitudes can
experience these conditions several days a month for two to three months a year. See Part Three for further
information on the description of the environments.
b. Cyclic high humidity (Cycle B2). Cyclic high humidity conditions are found in the open in tropical areas
where solar radiation is a factor. If the item in its operational configuration is subject to direct solar radiation
exposure, it is permissible to conduct the natural cycle with simulated solar radiation. See Part Three, Table
VII, for the associated B2 diurnal solar radiation parameters. In these areas, exposed items are subject to
alternate wetting and drying, but the frequency and duration of occurrence are essentially the same as in the
constant high humidity areas. Cycle B2 conditions occur in the same geographical areas as the Cycle B1
conditions, but the B1 conditions typically are encountered under a jungle canopy, so the B1 description
above also applies to the B2 area.
c. Hot-humid (Cycle B3). Severe (high) dewpoint conditions occur 10 to 15 times a year along a very narrow
coastal strip, probably less than 5 miles wide, bordering bodies of water with high surface temperatures,
specifically the Persian Gulf and the Red Sea. If the item in its operational configuration is subject to direct
solar radiation exposure, it is permissible to conduct the natural cycle with simulated solar radiation. See Part
Three, Table V for the associated B3 diurnal solar radiation parameters. Most of the year, these same areas
experience hot dry (A1) conditions. This cycle is unique to materiel to be deployed specifically in the Persian
Gulf or Red Sea regions, and is not to be used as a substitute for worldwide exposure requirements where B1
or B2 would apply.
In addition to these three categories of natural high-humidity conditions, there are three cycles for induced (storage and
transit) conditions:
d. Induced constant high humidity (Cycle B1). Relative humidity above 95 percent in association with nearly
constant 27 °C (80 °F) temperature occurs for periods of a day or more.
e. Induced variable - high humidity (Cycle B2). This condition exists when materiel in the
variable-high-humidity category receives heat from solar radiation with little or no cooling air. See storage
and transit conditions associated with the hot-humid daily cycle of the hot climatic design type below in
Table 507.6-I.
f. Induced hot-humid (Cycle B3). This condition exists when materiel in the hot-humid category receives heat
from solar radiation with little or no cooling air. The daily cycle for storage and transit in Table 507.6-I
shows 5 continuous hours with air temperatures at or above 66 °C (150 °F), and an extreme air temperature
of 71 °C (160 °F) for not more than 1 hour.
507.6-4

METHOD 507.6
NOTE: The climate station selected for these categories was Majuro, Marshall Islands (7o05’ N,
171o23’E). The station is located at the Majuro Airport Weather Services building. This site is a first-
order U.S. weather reporting station. Majuro was selected over 12 available candidate stations from
around the world initially because it possessed the required temperature and precipitation characteristics
for the B1 category (resulting in high temperature – humidity conditions), and it met the criteria for data
availability and quality.
On the average, Majuro receives over 3,300 mm (130 inches) of rainfall annually. Over 250 days
experience rainfall >= 0.01 inch, and over 310 days experience rainfall >= trace. Ten years of continuous
data were used for the analysis (POR: 1973-1982).
Groupings of consecutive days of rainfall (and resulting humidity) were then extracted. The longest
continuous streak of consecutive days >= trace was 51. A cumulative frequency curve was then created.
The recommended duration value of 45 days represents the 99th percentile value (actual value = 98.64%).
NOTE: During or after this test, document any degradation that could contribute to failure of the test
item during more extensive exposure periods (i.e., indications of potential long term problems), or
during exposure to other deployment environments such as shock and vibration. Further, extend testing
for a sufficient period of time to evaluate the long-term effect of its realistic deployment duration
(deterioration rate becomes asymptotic).
a. Procedure Ia - Induced (Storage and Transit) Cycles
(1) Hazardous test items. Hazardous test items will generally require longer tests than nonhazardous items to
establish confidence in test results. Since induced conditions are much more severe than natural
conditions, potential problems associated with high temperature/high relative humidity will be revealed
sooner, and the results can be analyzed with a higher degree of confidence. Consequently, expose
hazardous test items to extended periods (double the normal periods) of conditioning, depending upon the
geographical category to which the materiel will be exposed (see Table 507.6-II, induced cycles B1
through B3).
(2) Non-hazardous test items. Induced conditions are much more severe than natural conditions, and
potential problems associated with high temperature/high humidity will thus be revealed sooner, and the
results can be analyzed, in most cases, with a higher degree of confidence. Expose non-hazardous test
items to test durations as specified in Table 507.6-II, induced cycles B1 through B3, depending upon the
geographical category to which the material will be exposed.
b. Procedure Ib - Natural Cycles
(1) Hazardous test items. Hazardous test items are those in which any unknown physical deterioration
sustained during testing could ultimately result in damage to materiel or injury or death to personnel
when the test item is used. Hazardous test items will generally require longer test durations than
nonhazardous test items to establish confidence in the test results. Twice the normal test duration is
recommended (see Table 507.6-II, cycles B1 through B3).
(2) Nonhazardous test items. Nonhazardous test items should be exposed from 15 to 45 cycles of
conditioning, depending upon the geographical area to which the materiel will be exposed (see Table
507.6-II, cycles B1 through B3).
507.6-6

METHOD 507.6
(2) If an operational test procedure is required following the Induced (Storage and Transit) test.
(3) Periods of materiel operation or designated times for visual examinations (see paragraph 2.4.1).
(4) Operating test procedures, if appropriate.
c. Tailoring. Necessary variations in the basic test procedures to accommodate environments identified in the
LCEP.
3.2 During Test.
Collect the following information during conduct of the test:
a. General. Information listed in Part One, paragraph 5.10, and in Annex A, Tasks 405 and 406 of this
Standard.
b. Specific to this Method.
(1) Record of chamber temperature and humidity versus time conditions.
(2) Test item performance data and time/duration of checks.
3.3 Post-Test.
The following post test data shall be included in the test report.
a. General. Information listed in Part One, paragraph 5.13, and in Annex A, Task 406 of this Standard.
b. Specific to this Method.
(1) Previous test methods to which the test item has been subjected.
(2) Results of each operational check (before, during, and after test) and visual examination (and
photographs, if applicable).
(3) Length of time required for each operational check.
(4) Exposure durations and/or number of test cycles.
(5) Test item configuration and special test setup provisions.
(6) Any deviation from published cycles / procedures.
(7) Any deviations from the original test plan.
4. TEST PROCESS.
4.1 Test Facility.
Ensure the apparatus used in performing the humidity test includes the following:
4.1.1 General Description.
The required apparatus consists of a chamber or cabinet, and auxiliary instrumentation capable of maintaining and
monitoring (see Part One, paragraph 5.18) the required conditions of temperature and relative humidity throughout
an envelope of air surrounding the test item. (See Part One, paragraph 5.)
4.1.2 Facility Design.
Unless otherwise specified, use a test chamber or cabinet with a test volume and the accessories contained therein
constructed and arranged in such a manner as to prevent condensate from dripping on the test item. Vent the test
volume to the atmosphere to prevent the buildup of total pressure and prevent contamination from entering.
4.1.3 Test Sensors and Measurements.
Determine the relative humidity by employing either solid state sensors whose calibration is not affected by water
condensation, or by an equivalent method such as fast-reacting wet-bulb/dry-bulb sensors or dew point indicators.
Sensors that are sensitive to condensation, such as the lithium chloride type, are not recommended for tests with
high relative humidity levels. A data collection system, including an appropriate recording device(s), separate from
the chamber controllers is necessary to measure test volume conditions. If charts are used, use charts readable to
507.6-15

METHOD 507.6
within ±0.6 °C (±1 °F). If the wet-wick control method is approved for use, clean the wet bulb and tank and install a
new wick before each test and at least every 30 days. Ensure the wick is as thin as realistically possible to facilitate
evaporation (approximately 1/16 in. thick) consistent with maintaining a wet surface around the sensor. Use water
in wet-wick systems that is of the same quality as that used to produce the humidity (see Part One, paragraph 5.16).
When physically possible, visually examine the water bottle, wick, sensor, and other components making up relative
humidity measuring systems at least once every 24 hours during the test to ensure they are functioning as desired.
4.1.4 Air Velocity.
Use an air velocity flowing across the wet bulb sensor of not less than 4.6 meters/second (900 feet/minute, or as
otherwise specified in sensor response data), and ensure the wet wick is on the suction side of the fan to eliminate
the effect of fan heat. Maintain the flow of air anywhere within the envelope of air surrounding the test item
between 0.5 and 1.7 meters/second (98 to 335 feet/minute).
4.1.5 Humidity Generation.
Use steam or water injection to create the relative humidity within the envelope of air surrounding the test item. Use
water as described in Part One, paragraph 5.16. Verify its quality at periodic intervals (not to exceed 15 days) to
ensure its acceptability. If water injection is used to humidify the envelope of air, temperature-condition it before its
injection to prevent upset of the test conditions, and do not inject it directly into the test section. From the test
volume, drain and discard any condensate developed within the chamber during the test so as to not reuse the water.
4.1.6 Contamination Prevention.
Do not bring any material other than water into physical contact with the test item(s) that could cause the test item(s)
to deteriorate or otherwise affect the test results. Do not introduce any rust or corrosive contaminants or any
material other than water into the chamber test volume. Achieve dehumidification, humidification, heating and
cooling of the air envelope surrounding the test item by methods that do not change the chemical composition of the
air, water, or water vapor within that volume of air.
4.2 Controls.
a. Measurement and recording device(s). Ensure the test chamber includes an appropriate measurement and
recording device(s), separate from the chamber controllers.
b. Test parameters. Unless otherwise specified, make continuous analog temperature and relative humidity
measurements during the test. Conduct digital measurements at intervals of 15 minutes or less.
c. Capabilities. Use only instrumentation with the selected test chamber that meets the accuracies, tolerances,
etc., of Part One, paragraph 5.3.
4.3 Test Interruption.
Test interruptions can result from two or more situations, one being from failure or malfunction of test chambers or
associated laboratory test equipment. The second type of test interruption results from failure or malfunction of the
test item itself during operational checks.
4.3.1 Interruption Due to Chamber Malfunction.
a. General. See Part One, paragraph 5.11, of this Standard.
b. Specific to this Method.
(1) Undertest interruption. If an unscheduled interruption occurs that causes the test conditions to fall
below allowable limits, the test must be reinitiated at the end of the last successfully completed cycle.
(2) Overtest interruptions. If the test item(s) is exposed to test conditions that exceed allowable limits,
conduct an appropriate physical examination of the test item and perform an operational check (when
practical) before testing is resumed. This is especially true where a safety condition could exist, such
as with munitions. If a safety condition is discovered, the preferable course of action is to terminate
the test and reinitiate testing with a new test item. If this is not done and test item failure occurs during
the remainder of the test, the test results may be considered invalid. If no problem has been
encountered during the operational checkout or the visual inspection, reestablish pre-interruption
507.6-16

METHOD 507.6
conditions and continue from the point where the test tolerances were exceeded. See paragraph 4.3.2
for test item operational failure guidance.
4.3.2 Interruption Due to Test Item Operation Failure.
Failure of the test item(s) to function as required during operational checks presents a situation with several possible
options.
a. The preferable option is to replace the test item with a “new” one and restart from Step 1.
b. A second option is to replace / repair the failed or non-functioning component or assembly with one that
functions as intended, and restart the entire test from Step 1.
NOTE: When evaluating failure interruptions, consider prior testing on the same test
item and consequences of such.
4.4 Test Execution.
The following steps, alone or in combination, provide the basis for collecting necessary information concerning the
test item in a warm, humid environment.
4.4.1 Preparation for Test.
4.4.1.1 Test Setup.
a. General. See Part One, paragraph 5.8.
b. Unique to this Method. Verify that environmental monitoring and measurement sensors are of an
appropriate type and properly located to obtain the required test data.
4.4.1.2 Preliminary Steps.
Before starting the test, determine the test details (e.g., procedure variations, test item configuration, cycles,
durations, parameter levels for storage/operation, etc.) from the test plan.
4.4.1.3 Pretest Checkout.
All items require a pretest checkout at standard ambient conditions to provide baseline data. Conduct the checkout
as follows:
Step 1. Install appropriate instrumentation, e.g., thermocouples, in or on the test item.
Step 2. Install the test item into the test chamber and prepare the test item in its storage and/or transit
configuration in accordance with Part One, paragraph 5.8.1.
Step 3. Conduct a thorough visual examination of the test item to look for conditions that could
compromise subsequent test results.
Step 4. Document any significant results.
Step 5. Conduct an operational checkout (if appropriate) in accordance with the test plan, and record
results.
Step 6. If the test item operates satisfactorily, proceed to the appropriate test procedure. If not, resolve the
problems and repeat Step 5 above.
4.4.2 Test Procedures.
4.4.2.1 Procedure I - Storage and Transit Cycles (Cycles B2 or B3), and Natural (Cycles B1, B2, or B3).
Step 1. With the test item in the chamber, ensure it is in its storage and/or transit configuration, adjust the
chamber temperature to 23 ± 2 °C (73 ± 3.6 °F) and 50 ± 5 percent RH, and maintain for no less
than 24 hours.
Step 2. Adjust the chamber temperature and relative humidity to those shown in the appropriate induced
(storage and transit) category of Table 507.6-I for time 2400.
507.6-17

METHOD 507.6
Step 3. Unless other guidance is provided by the test plan, cycle the chamber air temperature and RH with
time as shown in the appropriate storage and transit cycle of Table 507.6-I (or in the appropriate
approximated curve from Figures 507.6-1, 507.6-2, or 507.6-3) through the 24-hour cycle, and for
the number of cycles indicated in Table 507.6-II for the appropriate climatic category.
Step 4. Adjust the chamber temperature to 23 ± 2 °C (73 ± 3.6 °F) and 50 ± 5 percent RH, and maintain
until the test item has reached temperature stabilization (generally not more than 24-hours).
Step 5. If only a storage and/or transit test is required, go to Step 15.
Step 6. Conduct a complete visual checkout of the test item and document the results.
Step 7. Put the test item in its normal operating configuration.
Step 8. Conduct a complete operational checkout of the test item and document the results. If the test item
fails to operate as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Otherwise, go to Step 9.
Step 9. Compare these data with the pretest data.
Step 10. Adjust the test item configuration to that required for naturally occurring temperature humidity
cycles (B1, B2, or B3).
Step 11. Adjust the chamber conditions to those given in Table 507.6-I for the time 2400 of the specified
cycle.
Step 12. Perform 24-hour cycles for the number of cycles indicated in Table 507.6-II for the appropriate
climatic category with the time-temperature-humidity values specified in Table 507.6-I, or the
appropriate approximated curve of Figures 507.6-4 through 507.6-6.
Step 13. If the materiel (test item) could be functioning non-stop in the natural environment, operate the
test item continuously throughout the test procedure. If shorter operational periods are identified
in the requirements document(s), operate the test item at least once every five cycles, and during
the last cycle, for the duration necessary to verify proper operation. If the test item fails to operate
as intended, follow the guidance in paragraph 4.3.2 for test item failure.
Step 14. Adjust the chamber temperature to 23 ± 2 °C (73 ± 3.6 °F) and 50 ± 5 percent RH, and maintain
until the test item has reached temperature stabilization (generally not more than 24-hours).
Step 15. Conduct a complete visual examination of the test item and document the results.
Step 16. Conduct an operational checkout of the test item in accordance with the approved test plan, and
document the results. See paragraph 5 for analysis of results.
Step 17. Compare these data with the pretest data.
4.4.2.2 Procedure II - Aggravated.
This test consists of a 24-hour conditioning period (to ensure all items at any intended climatic test location will start
with the same conditions), followed by a series of 24-hour temperature and humidity cycles for a minimum of 10
cycles, or a greater number as otherwise specified in the test plan, unless premature facility or test item problems
arise.
Step 1. With the test item installed in the test chamber in its required configuration, adjust the temperature
to 23 ± 2 °C (73 ± 3.6 °F) and 50 ± 5 percent RH, and maintain for no less than 24 hours.
Step 2. Adjust the chamber temperature to 30 °C (86 °F) and the RH to 95 percent.
Step 3. Expose the test item(s) to at least ten 24-hour cycles ranging from 30-60 ºC (86-140 °F) (Figure
507.6-7) or as otherwise determined in paragraph 2.2.1. Unless otherwise specified in the test
plan, conduct a test item operational check (for the minimum time required to verify performance)
near the end of the fifth and tenth cycles during the periods shown in Figure 507.6-7, and
507.6-18

METHOD 507.6
document the results. If the test item fails to operate as intended, follow the guidance in paragraph
4.3.2 for test item failure. Otherwise, continue with Step 4.
NOTE: If the operational check requires the chamber to be open or the test item to be removed
from the chamber, and the check cannot be completed within 30 minutes, in order to prevent
unrealistic drying, recondition the test item at 30°C ±2 °C (86°F ± 4 °F) and 95 percent RH for
one hour, and then continue the checkout. Extend the test time for that cycle by one hour.
Continue this sequence until the checkout has been completed.
If the operational check is conducted in the chamber, and extends beyond the 4-hour period noted
in Figure 507.6-7, do not proceed to the next cycle until the checkout is completed. Once the
check has been completed resume the test.
Step 4. At the completion of 10 or more successful cycles, adjust the temperature and humidity to 23
±2 °C (73 ± 3.6 °F) and 50 ± 5 percent RH, and maintain until the test item has reached temperature
stabilization (generally not more than 24-hours).
Step 5. Perform a thorough visual examination of the test item, and document any conditions resulting
from test exposure.
Step 6. Conduct a complete operational checkout of the test item and document the results. See paragraph 5
for analysis of results.
Step 7. Compare these data with the pretest data.
5. ANALYSIS OF RESULTS.
In addition to the guidance provided in Part One, paragraphs 5.14 and 5.17, the following information is provided to
assist in the evaluation of the test results.
a. Allowable or acceptable degradation in operating characteristics.
b. Possible contributions from special operating procedures or special test provisions needed to perform
testing.
c. Whether it is appropriate to separate temperature effects from humidity effects.
d. Any deviations from the test plan.
6. REFERENCE/RELATED DOCUMENTS.
6.1 Referenced Documents.
a. AR 70-38, Research, Development, Test and Evaluation of Materiel for Extreme Climatic Conditions;
1 August 1979.
b. MIL-HDBK-310, Global Climatic Data for Developing Military Products; 23 June 1997.
c. NATO STANAG 4370, Allied Environmental Conditions and Test Publication (AECTP) 230; Climatic
Conditions.
d. NATO STANAG 4370, Environmental Testing; 19 April 2005.
e. Allied Environmental Conditions and Test Publication (AECTP) 200, Environmental Conditions (under
STANAG 4370), January 2006
6.2 Related Documents.
a. Synopsis of Background Material for MIL-STD-210B, Climatic Extremes for Military Equipment.
Bedford, MA: Air Force Cambridge Research Laboratories, 24 January 1974. DTIC number
AD-780-508.
b. Allied Environmental Conditions and Test Publication (AECTP) 300, Climatic Environmental Tests (under
STANAG 4370), Method 306; January 2006.
507.6-19

METHOD 507.6
c. Egbert, Herbert W. “The History and Rationale of MIL-STD-810 (Edition 2),” January 2010; Institute of
Environmental Sciences and Technology, Arlington Place One, 2340 S. Arlington Heights Road, Suite 100,
Arlington Heights, IL 60005-4516.
(Copies of Department of Defense Specifications, Standards, and Handbooks, and International Standardization
Agreements are available online at https://assist.dla.mil.
Requests for other defense-related technical publications may be directed to the Defense Technical Information
Center (DTIC), ATTN: DTIC-BR, Suite 0944, 8725 John J. Kingman Road, Fort Belvoir VA 22060-6218,
1-800-225-3842 (Assistance--selection 3, option 2), http://www.dtic.mil/dtic/; and the National Technical
Information Service (NTIS), Springfield VA 22161, 1-800-553-NTIS (6847), http://wwwntis.gov/.
507.6-20

METHOD 507.6, ANNEX A
METHOD 507.6, ANNEX A
PHYSICAL PHENOMENA ASSOCIATED WITH HUMIDITY
1. ABSORPTION.
The accumulation of water molecules within material. The quantity of water absorbed depends, in part, on the water
content of the ambient air. The process of absorption occurs continuously until equilibrium is reached. The
penetration speed of the molecules in the water increases with temperature.
2. ADSORPTION.
The adherence of water vapor molecules to a surface whose temperature is higher than the dew point. The quantity
of moisture that can adhere to the surface depends on the type of material, its surface condition, and the vapor
pressure. An estimation of the effects due solely to adsorption is not an easy matter because the effects of
absorption, that occurs at the same time, are generally more pronounced.
3. BREATHING.
Air exchange between a hollow space and its surroundings caused by temperature variations. This commonly
induces condensation inside the hollow space.
4. CONDENSATION.
The precipitation of water vapor on a surface whose temperature is lower than the dew point of the ambient air. As
a consequence, the water is transformed from the vapor state to the liquid state.
The dew point depends on the quantity of water vapor in the air. The dew point, the absolute humidity, and the
vapor pressure are directly interdependent. Condensation occurs on a test item when the temperature at the surface
of the item placed in the test chamber is lower than the dew point of the air in the chamber. As a result, the item
may need to be preheated to prevent condensation.
Generally speaking, condensation can only be detected with certainty by visual inspection. This, however, is not
always possible, particularly with small objects having a rough surface. If the test item has a low thermal constant,
condensation can only occur if the air temperature increases abruptly, or if the relative humidity is close to 100
percent. Slight condensation may be observed on the inside surface of box structures resulting from a decrease in
the ambient temperature.
5. DIFFUSION.
The movement of water molecules through material caused by a difference in partial pressures. An example of
diffusion often encountered in electronics is the penetration of water vapor through organic coatings such as those
on capacitors or semiconductors, or through the sealing compound in the box.
507.6A-1

METHOD 507.6, ANNEX A
(This page is intentionally blank.)
507.6A-2

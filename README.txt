Raceline generator: Based on code by YouTuber NeuralNine

Computer setup:
-Install python from python.org/downloads
-Make sure to install pip during this process
-For the code to run, the following libraries will need to be installed:
	-Opencv-python
	-Neat-python
	-Pygame
	-Matplotlib
	This can be done by typing "pip install [library]" into command prompt 

Required files:
-Map file (sample file vegas.png is included)
-car.png
-config.txt
-newcar.py
-All above files should be in the same folder.

Map file constants:
-BORDER_COLOR: Colour of non-track sections of the map. By default, the track can be any colour except white (#000000).
-START_COLOR and START_CV2: Colour of start line. By default, the start line must be green (#00FF00).
-FILE_TYPE: File type of map file. By default, this is .png, but any image file should be readable.

Other constants:
-CAR_SIZE_X and CAR_SIZE_Y: Used to change size of car so that it can be relative to track size.
-DIRECTION: Direction of car movement. Counterclockwise (0) by default, switch to 180 for clockwise driving.
-NUM_LAPS: Number of laps that will occur before raceline generation.

Running the code:
-When asked for the file name, write it in the form "[name].png"
-The raceline will be generated once the first car completes two laps. Close the window to simulate the next generation.
-If all cars die before the first car completes two laps, no raceline will be generated and the next generation will be simulated.
-The generation time displayed represents the time up to the point where the raceline window is closed. To use an accurate representation of generation time, comment out the "get_raceline(nets, cars, best_car, game_map, im)" code in the run_simulation function.

Reading speed:
-Speed range and starting speed will be displayed prior to raceline generation.
-Changes in colour indicate a change in speed:
	Red -> Green -> Blue -> Red indicates an increase in speed
	Red -> Blue -> Green -> Red indicates a decrease in speed
-Fluctuations between two speeds (ex. Red -> Green -> Red -> Green) are common.
-Speeds are relative, their physical values have no meaning.

Known bugs:
-One lap will occasionally be counted twice, resulting in the raceline not appearing.
-Cars can die during the additional lap, resulting in an incomplete raceline.
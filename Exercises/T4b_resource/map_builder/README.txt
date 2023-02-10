=== Running the builder ===
Map builder allows creating custom maps to test your code.

Run it using builder_main.py where you can specify size of the grid using x_size and y_size. If the whole GUI is too large
you can change the size of the buttons in GUI.py by changing constant BUTTON_SIZE

=== Controlling the builder ===
When clicking the grid it will place wall, entrance or goal based on current selection. You can overwrite already placed objects.
Default selection when builder starts is placing walls.

Clicking the BLACK button sets the current selection to placing WALLS.
Clicking the GREEN button sets the current selection to placing ENTRANCES. (note that entrances must be on the edge of the map)
Clicking the YELLOW button sets the current selection to placing GOALS.

Clicking the RED button starts placing a sensor which will mark the selected cells red. When you select all the cells of the sensor, click the RED button again
and the sensor is saved and the current selection will be none. Repeat to place more sensors.

Clicking the BLUE button will save the map to new_map in the map_builder folder. You can change the path in map_model.py using constant PATH.
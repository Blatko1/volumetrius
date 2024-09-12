# Volumetrius

## IDEAS

- when two object's bounding boxes are intersected, I have to check them both when drawing, because if the further bb has a part in front of some segments of the closer bb, the closer one would still be drawn. I need a depth buffer for this. 
So the steps are:
* a ray goes forward and hits the big bb first
* check if the big bb intersects with some other bb,
* if not, continue normally,
* if yes, first draw the closer object (bb) and remember the distance to pixel, then try to find pixel of second closest (bb) object and compare pixel distances, and so on
* after all that, you end up with proper depth
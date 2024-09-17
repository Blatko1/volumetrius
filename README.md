# Volumetrius

## Idea and Goal

Primary goal is to recreate Minecraft but it's completely ray traced (or path traced (idk)). So NO triangles (except the triangle graphics are being drawn on)!

Some basic objects are chunks with dimensions 32x32x32 or 64x64x64 (TBD), some entities and custom made voxel objects, lightning, maybe terrain generation after all GPU rendering is situated.

Chunks and objects are optimized into a SVO each.

## IDEAS

- when two object's bounding boxes are intersected, I have to check them both when drawing, because if the further bb has a part in front of some segments of the closer bb, the closer one would still be drawn. I need a depth buffer for this. 
So the steps are:
* a ray goes forward and hits the big bb first
* check if the big bb intersects with some other bb,
* if not, continue normally,
* if yes, first draw the closer object (bb) and remember the distance to pixel, then try to find pixel of second closest (bb) object and compare pixel distances, and so on
* after all that, you end up with proper depth

- How to transfer SVOs to GPU???
* Flatten all SVOs so that each becomes a ray of Nodes. Each Node has index to 8 children, a parent and a bool wether it is the last node (or make children indices negative if the child is a leaf).
* Into a uniform buffer send an array of indices for each SVO where and index represents the starting point in the SVO array of a unique SVO. 

- How to traverse a SVO???
* idk, yet

## TODO

- [x] Objects are able to rotate
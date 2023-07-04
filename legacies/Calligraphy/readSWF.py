from swf.movie import SWF

# create a file object
file = open('path/to/swf', 'rb')

# print out the SWF file structure
print (SWF(file))
'''

from swf.movie import SWF
from swf.export import SVGExporter

file = open('StrokeExercisePanel.swf', 'rb')

swf = SWF(file)

svg_exporter= SVGExporter()

svg = swf.export(svg_exporter)

open('item.svg', 'wb').write(svg.read())

import cairo
import rsvg

img = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640,480)

ctx = cairo.Context(img)

## handle = rsvg.Handle(<svg filename>)
# or, for in memory SVG data:
# handle= rsvg.Handle(None, str(<svg data>))
handle= rsvg.Handle(None, svg)

handle.render_cairo(ctx)

img.write_to_png("svg.png")
'''
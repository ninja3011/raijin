# KLayout batch render of a sky130 GDS.
# klayout -zz -r render_layout.py -rd gds=... -rd lyp=... -rd out=... [-rd w=2400] [-rd box=x1,y1,x2,y2 (um)] [-rd layers=full|zoom]
import pya
w = int(globals().get("w", "2400"))
mode = globals().get("layers", "full")
# drawing layers kept visible (GDS layer/datatype); everything else (wells, area ids, boundaries, fill markers) hidden
KEEP_FULL = {(68,20),(69,20),(70,20),(71,20),(72,20)}                 # met1..met5
KEEP_ZOOM = {(65,20),(66,20),(67,20),(67,44),(68,20),(68,44),(69,20),(69,44),(70,20),(70,44),(71,20)}
keep = KEEP_FULL if mode == "full" else KEEP_ZOOM
lv = pya.LayoutView()
lv.set_config("background-color", "#ffffff")
lv.set_config("grid-visible", "false")
lv.set_config("text-visible", "false")
ci = lv.load_layout(gds, True)
lv.load_layer_props(lyp)
lv.max_hier()
it = lv.begin_layers()
while not it.at_end():
    lp = it.current()
    on = (lp.source_layer, lp.source_datatype) in keep
    if lp.visible != on:
        lp.visible = on
        lv.set_layer_properties(it, lp)
    it.next()
top = lv.cellview(ci).cell
bb = top.dbbox()
if globals().get("box"):
    x1, y1, x2, y2 = [float(v) for v in box.split(",")]
    bb = pya.DBox(x1, y1, x2, y2)
lv.zoom_box(bb)
h = int(w * bb.height() / bb.width())
lv.save_image(out, w, h)
print("wrote", out, w, h, "bbox", bb)

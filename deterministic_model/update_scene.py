def make_json_serializable(obj):
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(x) for x in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj
    
def find_wire(wire_in):
    stage = omni.usd.get_context().get_stage()
    for w, prim_wire in enumerate(stage.Traverse()):
        prim_wire_name = prim_wire.GetName()
        if prim_wire_name == wire_in["name"]:
            wire_position = find_position(prim_wire.GetPath())
            wire_position = make_json_serializable(wire_position)
    return wire_position

def find_terminal(term_in):
    stage = omni.usd.get_context().get_stage()
    for prim_term in stage.Traverse():
        prim_term_name = prim_term.GetName()
        if prim_term_name == term_in["name"]:
            term_position = find_position(prim_term.GetPath())
            term_position = make_json_serializable(term_position)
    return term_position

def update_file(self, wire_in, term_in):
        self.find_wire(wire_in)
        self.find_terminal(term_in)
        return self.vision_out
        
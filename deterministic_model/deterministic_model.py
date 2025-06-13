class DeterministicActionModel:
    def __init__(self, target_wire, manipulated_object=False):
        
        self.target_wire_id = target_wire["ID"]
        self.target_wire_name = target_wire["name"]
        self.target_wire_state = target_wire["state"]
        
        if manipulated_object != None:
            self.manipulated_object = True
        
    def map_function(self):
        if self.target_wire_state == 'held':
            return 'insert'
        elif self.target_wire_state == 'inserted':
            return 'lock'
        elif self.target_wire_state == 'on_table':
            if self.manipulated_object:
                return 'putdown'
            else:
                return 'pick'
        else:
            return 'unknown'

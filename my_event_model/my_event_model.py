import hashlib
import numpy as np

class Event(object):
    '''Event defined by its json description.'''
    def __init__(self, json_data):
        self.number_of_modules = len(json_data["module_prefix_sum"]) - 1
        print("Event has {} modules.".format(self.number_of_modules))
        self.description = json_data.get("description", "")
        self.montecarlo = json_data.get("montecarlo", "")
        self.module_prefix_sum = json_data["module_prefix_sum"]
        self.number_of_hits = self.module_prefix_sum[self.number_of_modules]
        self.hits = []
        self.with_t = "t" in json_data

        self.modules = []
        for m in range(self.number_of_modules):
            start_idx = self.module_prefix_sum[m]
            end_idx = self.module_prefix_sum[m + 1]
            module_hits = []
            z_values = set()
            for i in range(start_idx, end_idx):
                if self.with_t:
                    new_hit = Hit(
                        x=json_data["x"][i],
                        y=json_data["y"][i],
                        z=json_data["z"][i],
                        hit_id=i,
                        module=m,
                        t=json_data["t"][i],
                        with_t=True
                    )
                else:
                    new_hit = Hit(
                        x=json_data["x"][i],
                        y=json_data["y"][i],
                        z=json_data["z"][i],
                        hit_id=i,
                        module=m
                    )
                self.hits.append(new_hit)
                module_hits.append(new_hit)
                z_values.add(json_data["z"][i])
            # Assuming each module has a single z value
            module_z = z_values.pop() if len(z_values) == 1 else np.mean(list(z_values))
            new_module = Module(
                module_number=m,
                z=module_z,
                hits=module_hits
            )
            self.modules.append(new_module)

        # Print the total number of hits
        print("Total number of hits in event:", len(self.hits))
        # Print the number of modules created
        print("Total number of modules in event:", len(self.modules))

    def compute_hamiltonian(self, hamiltonian_function, params):
        '''Compute the Hamiltonian for the event using the provided hamiltonian_function.
        
        The hamiltonian_function should accept the event and params as arguments.
        '''
        return hamiltonian_function(self, params)

class Track(object):
    '''A track, essentially a list of hits.'''
    def __init__(self, hits):
        self.hits = hits
        self.missed_last_module = False
        self.missed_penultimate_module = False

    def add_hit(self, hit):
        self.hits.append(hit)

    def __repr__(self):
        return "Track with " + str(len(self.hits)) + " hits: " + str(self.hits)

    def __iter__(self):
        return iter(self.hits)

    def __eq__(self, other):
        return self.hits == other.hits

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return int.from_bytes(hashlib.sha256(
            ''.join([str(h.id) for h in self.hits]).encode('utf-8')).digest(), byteorder='big')
    
class track(object):
 '''A track, essentially a list of hits.'''
 def __init__(self, hits):
   self.hits = hits
   self.missed_last_module = False
   self.missed_penultimate_module = False

 def add_hit(self, hit):
   self.hits.append(hit)
 
 def get_hits(self):
   hit_list = []
   for hit in self.hits:
     hit_list.append([hit.x,hit.y,hit.z])
   return hit_list

 def __repr__(self):
   return "Track with " + str(len(self.hits)) + " hits: " + str(self.hits)

 def __iter__(self):
   return iter(self.hits)

 def __eq__(self, other):
   return self.hits == other.hits

 def __ne__(self, other):
   return not self.__eq__(other)

 def __hash__(self):
     return int.from_bytes(hashlib.sha256(
       ''.join([str(h.id) for h in self.hits]).encode('utf-8')).digest(), byteorder='big')

class Hit(object):
    '''A hit, composed of an id and its x, y, and z coordinates.
    It may optionally contain the number of the module where
    the hit happened.
    '''
    def __init__(self, x, y, z, hit_id, module=-1, t=0, with_t=False):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.id = hit_id
        self.module_number = module
        self.module_id = module  # Alias for module_number
        self.with_t = with_t

    def __getitem__(self, index):
        if index < 0 or index > 2:
            raise IndexError
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        else:
            return self.z

    def __repr__(self):
        return "#" + str(self.id) + " module " + str(self.module_number) + \
            " {" + str(self.x) + ", " + \
            str(self.y) + ", " + str(self.z) + \
            (", " + str(self.t) if self.with_t else "") + "}"

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.id

class Module(object):
    '''A module is identified by its number.
    It also contains the z coordinate where it sits and
    the list of hits it holds.
    '''
    def __init__(self, module_number, z, hits):
        self.module_number = int(module_number)
        self.z = z  # Single z value
        self.hits_list = hits

    def __iter__(self):
        return iter(self.hits())

    def __repr__(self):
        return "module " + str(self.module_number) + ":\n" + \
            " At z: " + str(self.z) + "\n" + \
            " Number of hits: " + str(len(self.hits())) + "\n" + \
            " Hits (#id {x, y, z}): " + str(self.hits())

    def hits(self):
        return self.hits_list

class Segment(object):
    '''Represents a segment between two hits.'''
    def __init__(self, from_hit, to_hit):
        self.from_hit = from_hit
        self.to_hit = to_hit

    def to_vect(self):
        '''Returns the vector from from_hit to to_hit.'''
        return np.array([
            self.to_hit.x - self.from_hit.x,
            self.to_hit.y - self.from_hit.y,
            self.to_hit.z - self.from_hit.z
        ])

    def __eq__(self, other):
        return self.from_hit == other.from_hit and self.to_hit == other.to_hit

    def __hash__(self):
        return hash((self.from_hit, self.to_hit))

    def __repr__(self):
        return f"Segment(from_hit={self.from_hit.id}, to_hit={self.to_hit.id})"
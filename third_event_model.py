import dataclasses
from mpl_toolkits.mplot3d import Axes3D, art3d
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.animation as animation

def set_axes_equal(ax):
   '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
   cubes as cubes, etc..  This is one possible solution to Matplotlib's
   ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

   Input
     ax: a matplotlib axis, e.g., as output from plt.gca().
   '''

   x_limits = ax.get_xlim3d()
   y_limits = ax.get_ylim3d()
   z_limits = ax.get_zlim3d()

   x_range = abs(x_limits[1] - x_limits[0])
   x_middle = np.mean(x_limits)
   y_range = abs(y_limits[1] - y_limits[0])
   y_middle = np.mean(y_limits)
   z_range = abs(z_limits[1] - z_limits[0])
   z_middle = np.mean(z_limits)

   # The plot bounding box is a sphere in the sense of the infinity
   # norm, hence I call half the max range the plot radius.
   plot_radius = 0.5*max([x_range, y_range, z_range])

   ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
   ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
   ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

@dataclasses.dataclass(frozen=True) 
class Hit:
   hit_id: int
   x: float
   y: float
   z: float
   module_id: int
   track_id: int
   hit_id: int

   def __getitem__(self, index):
       return (self.x, self.y, self.z)[index]
   
   def __eq__(self, __value: object) -> bool:
       return self is __value
       #if self.hit_id == __value.hit_id:
       #    return True
       #else:
       #    return False

@dataclasses.dataclass(frozen=True)
class Module:
   module_id: int
   z: float
   lx: float
   ly: float
   hits: list[Hit]
   
   def __eq__(self, __value: object) -> bool:
       if self.module_id == __value.module_id:
           return True
       else:
           return False

@dataclasses.dataclass
class MCInfo:
   primary_vertex  : tuple
   theta           : float
   phi             : float
   
   
@dataclasses.dataclass(frozen=True)
class Track:
   track_id: int
   mc_info : MCInfo
   hits    : list[Hit]
   
   def __eq__(self, __value: object) -> bool:
       return self is __value
       #if self.track_id == __value.track_id:
       #    return True
       #else:
       #    return False

@dataclasses.dataclass(frozen=True)
class Event:
   modules: list[Module]
   tracks: list[Track]
   hits: list[Hit]

   def display(self, ax: Axes3D, show_tracks = True, show_hits = True, show_modules = True, equal_axis = True, s_hits=1):
       ax.set_facecolor('#CCCCCC')
       if show_hits:
           hit_x, hit_y, hit_z = [], [], []
           for hit in self.hits:
               hit_x.append(hit.x)
               hit_y.append(hit.y)
               hit_z.append(hit.z)
           ax.scatter3D(hit_x, hit_y, hit_z,s=s_hits,c='snow')

       if show_modules:
           for module in self.modules:
               p = Rectangle((-module.lx/2, -module.ly/2), module.lx, module.ly,alpha=.2,edgecolor='#20B2AA')
               ax.add_patch(p)
               art3d.pathpatch_2d_to_3d(p, z=module.z)
       
       if show_tracks:
           x_lim = ax.get_xlim()
           y_lim = ax.get_ylim()
           z_lim = ax.get_zlim()
           ts = []
           for track in self.tracks:
               pvx, pvy, pvz = track.mc_info.primary_vertex
               phi = track.mc_info.theta
               theta = track.mc_info.phi
               tx1 = max((x_lim[0] - pvx)/(np.sin(theta)*np.cos(phi)),0)
               tx2 = max((x_lim[1] - pvx)/(np.sin(theta)*np.cos(phi)),0)
               ty1 = max((y_lim[0] - pvy)/(np.sin(theta)*np.sin(phi)),0)
               ty2 = max((y_lim[1] - pvy)/(np.sin(theta)*np.sin(phi)),0)
               tz1 = max((z_lim[0] - pvz)/(np.cos(theta)),0)
               tz2 = max((z_lim[1] - pvz)/(np.cos(theta)),0)
               ts.append(min(max(tx1,tx2),max(ty1,ty2), max(tz1,tz2)))

           for track, t in zip(self.tracks, ts):
               pvx, pvy, pvz = track.mc_info.primary_vertex
               phi = track.mc_info.theta
               theta = track.mc_info.phi
               ax.plot((pvx,pvx + t*np.sin(theta)*np.cos(phi)),
               (pvy,pvy+ t*np.sin(theta)*np.sin(phi)),
               (pvz,z_lim[1]), alpha=1, color = '#BA82EB')

       if equal_axis:
           set_axes_equal(ax)
           ax.set_box_aspect([10,6,6])
       ax.set_proj_type('ortho')

@dataclasses.dataclass(frozen=True)
class Segment:
   segment_id  : int
   hit_from    : Hit
   hit_to      : Hit
   
   def __eq__(self, __value: object) -> bool:
       return self is __value
       #if self.segment_id == __value.segment_id:
       #    return True
       #else:
       #    return False
   
   
   def to_vect(self):
       return (self.hit_to.x - self.hit_from.x, 
               self.hit_to.y - self.hit_from.y, 
               self.hit_to.z - self.hit_from.z)
   
   
   def __mul__(self, __value):

       v_1 = self.to_vect()
       v_2 = __value.to_vect()
       n_1 = (v_1[0]**2 + v_1[1]**2 + v_1[2]**2)**0.5
       n_2 = (v_2[0]**2 + v_2[1]**2 + v_2[2]**2)**0.5
       
       return (v_1[0]*v_2[0] + v_1[1]*v_2[1] + v_1[2]*v_2[2])/(n_1*n_2)


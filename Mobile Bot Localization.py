import numpy as np
import math 
import matplotlib.pyplot as plt
import scipy.stats


"""U_Buildings = [[ 10.0, 20.0 ], # SSB
               [75,20],  #SC
               [ 100.0, 85.0 ], #HEDCO
               [ 100.0, 105.0 ], #MCE
               [ 110.0, 145.0 ], #WEB
               [165.0, 25.0 ], #SC
               [170.0, 50.0 ], #AH
               [ 125.0, 70.0 ], #ESB
               [20,70], #WBB
               [16,95], #FASB
               [25,125], #MEK
               [75,175]], #MEB"""

U_Buildings = [ [10.0, 20.0 ], 
               [75,20], 
               [ 100.0, 85.0 ], 
               [ 100.0, 105.0 ], 
               [ 110.0, 145.0 ], 
               [165.0, 25.0 ],
               [170.0, 50.0 ], 
               [ 125.0, 70.0 ], 
               [20,70], 
               [16,95], 
               [25,125], 
               [75,175]]

class mobilerobot:
    def __init__(self):
        #Randomly setting position of robot/particles in map
        self.x_coord = int(np.random.uniform(low=0.0, high=200))
        self.y_coord = int(np.random.uniform(low=0.0, high=200))
        self.orientation = np.random.uniform(low=0.0, high=2*math.pi)
        self.lidar_noise_variance = 15
        self.odometry_noise_variance = 0
        
    def set_robot_location(self,final_x,final_y,final_orientation,odometry_noise_variance):
        #Setting robot location after movement 
        self.x_coord =final_x
        self.y_coord = final_y
        self.orientation =final_orientation
        self.odometry_noise_variance = odometry_noise_variance
    
    def lidar(self):
        sense_vals = []
        
        for i in range (len(U_Buildings )):
            x_dist = self.x_coord - U_Buildings[i][0]
            y_dist = self.y_coord - U_Buildings[i][1]
            sense_val = math.sqrt((x_dist*x_dist) + (y_dist*y_dist)) +np.random.normal(0, self.lidar_noise_variance);
            sense_vals.append(sense_val)
        return sense_vals;
        

def pose_generator(initialrobot,distance,turningangle,odometry_noise_variance):
    #Making new robot(particle after resampling)
    noisy_distance = distance+ np.random.normal(0, initialrobot.odometry_noise_variance);
    noisy_turningangle = turningangle + np.random.normal(0, initialrobot.odometry_noise_variance);
    new_orient = initialrobot.orientation + turningangle;
    new_x = initialrobot.x_coord + (math.sin(turningangle) * distance) 
    new_y = initialrobot.y_coord + (math.cos(turningangle) * distance) 
    finalrobotpose = mobilerobot()
    finalrobotpose.set_robot_location(new_x,new_y,new_orient,odometry_noise_variance);
    return finalrobotpose



np.random.seed(123)
actualbot = mobilerobot()
actual_bot_lidar_measurements = actualbot.lidar()
plt.title('Particles as Red dots, Actual bot as Blue dot, Estimated bot as Green dot')
plt.xlim(0, 200);
plt.ylim(0, 200);
num_particles = 1000
particles =[]

for i in range(num_particles):
    particles.append(mobilerobot())
    
    
for i in range(num_particles):
    plt.plot(particles[i].x_coord,particles[i].y_coord, "ro",markersize=1);


#plt.plot(actualbot.x_coord,actualbot.y_coord,'ro')
#plt.show()
#input()

actualbot =pose_generator(actualbot,1.0,0.5,0)
actual_bot_lidar_measurements = actualbot.lidar()

particles2 =[]
odometry_noise_variance =0.1;

for i in range(num_particles):
    particles2.append(mobilerobot())
    particles2[i] = pose_generator(particles[i],1.0,0.5,odometry_noise_variance)


for i in range(len(U_Buildings)):
    mark_x = U_Buildings[i][0]
    mark_y = U_Buildings[i][1]
    plt.plot(mark_x, mark_y, "ko",markersize=6);
    
#comb_prob_dens =1.0
 
weights = []   #weights for each particle
for i in range(num_particles):
    comb_prob_dens =1.0
    particle_lidar_measurements =particles2[i].lidar()
    particle_sense_val =0.0
    mean=0.0
    
    for j in range (len(U_Buildings )):
        x_dist = particles2[i].x_coord - U_Buildings[j][0]
        y_dist = particles2[i].y_coord - U_Buildings[j][1]
        particle_sense_val = math.sqrt((x_dist*x_dist) + (y_dist*y_dist)) 
        mean = particle_sense_val  #Actual sense value without noise
        #mean = particle_lidar_measurements[j] 
        #(dist, particles[i].lidar_noise_variance, lidar_measurements[i]);
        #combining all landmarks prob dens
        #print(scipy.stats.norm(mean, particles[i].lidar_noise_variance).pdf(lidar_measurements[j]))
        prob = scipy.stats.norm.pdf(actual_bot_lidar_measurements[j],mean, particles2[i].lidar_noise_variance)
        comb_prob_dens = comb_prob_dens*prob
        #print(comb_prob_dens,prob,mean,actual_bot_lidar_measurements[j],particles2[i].lidar_noise_variance)
        
    weights.append(comb_prob_dens)
    #calculating weights for each particle

max_weight_idx = np.argmax(weights)  
plt.plot(actualbot.x_coord,actualbot.y_coord,'bo')
plt.plot(particles2[np.argmax(weights)].x_coord, particles2[np.argmax(weights)].y_coord,'go')
    
    

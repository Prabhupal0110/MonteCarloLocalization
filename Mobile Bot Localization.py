# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 13:06:53 2022

@author: Prabhupal Singh
"""


import numpy as np
import math 
import matplotlib.pyplot as plt
import scipy.stats

np.random.seed(65)


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
        self.lidar_noise_variance = 10
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
            sense_val = (math.sqrt((x_dist*x_dist) + (y_dist*y_dist)) + 
                        np.random.normal(0, self.lidar_noise_variance));
            sense_vals.append(sense_val)
        return sense_vals;
        

def pose_generator(initialrobot,distance,turningangle,odometry_noise_variance):
    #Making new robot(particle after resampling)
    noisy_distance = distance+ np.random.normal(0, odometry_noise_variance);
    noisy_turningangle = turningangle + np.random.normal(0, odometry_noise_variance);
    new_orient = initialrobot.orientation + noisy_turningangle;
    new_x = initialrobot.x_coord + (math.sin(noisy_turningangle) * noisy_distance) 
    new_x =  constrain(new_x, 200)
    new_y = initialrobot.y_coord + (math.cos(noisy_turningangle) * noisy_distance) 
    new_y =  constrain(new_y, 200)
    finalrobotpose = mobilerobot()
    finalrobotpose.set_robot_location(new_x,new_y,new_orient,odometry_noise_variance);
    return finalrobotpose

def constrain(first_term, second_term):
    #Constraining within map
    return int(first_term - (second_term)*np.floor(first_term / (second_term)));
    
def error_val(actualbot,particle):
    x_err = actualbot.x_coord -  particle.x_coord;
    y_err= actualbot.y_coord   - particle.y_coord;
    err = np.sqrt(x_err**2 + y_err**2)
    return (err)

def gaussian(mu, sigma, x):
   return np.exp(-(pow((mu - x), 2)) / (pow(sigma, 2)) / 2.0) / np.sqrt(2.0 * np.pi * (pow(sigma, 2)));


actualbot = mobilerobot()
actualbot.set_robot_location(25.0,75.0,actualbot.orientation,0)
actual_bot_lidar_measurements = actualbot.lidar()

num_particles = 1000
particles =[]

for i in range(0,num_particles):
    particles.append(mobilerobot())
    
actualbot =pose_generator(actualbot,1.0,np.pi/3,0)
actual_bot_lidar_measurements = actualbot.lidar()

particles2 =[]
odometry_noise_variance =0;

for i in range(0,num_particles):
    particles2.append(mobilerobot())
    particles2[i] = pose_generator(particles[i],1.0,np.pi/3,odometry_noise_variance)


fig = plt.figure(figsize= (10,10))
plt.xlim(0, 200);
plt.ylim(0, 200);


for k in range(0, len(particles2)):
    plt.plot(particles2[k].x_coord, particles2[k].y_coord,'ro',markersize=3)
    
plt.plot(actualbot.x_coord,actualbot.y_coord,'bo',markersize=10)  
for i in range(0,len(U_Buildings)):
    mark_x = U_Buildings[i][0]
    mark_y = U_Buildings[i][1]
    plt.plot(mark_x, mark_y, "ko",markersize=10);
    
#plt.title('Particles as Red dots, Actual bot as Blue dot, Estimated bot as Green dot')

Err_plot=[]
total_iterations=20
for c in range(total_iterations):
    print("Iteration: ",c+1);
    
    actualbot =pose_generator(actualbot,5.0,np.pi/2,0)
    actual_bot_lidar_measurements = actualbot.lidar()
    
    for i in range(0,num_particles):
        particles2[i] = pose_generator(particles2[i],5.0,np.pi/2,odometry_noise_variance)
    
     
    weights = []   #weights for each particle
    for i in range(0,num_particles):
        comb_prob_dens =1.0
        particle_lidar_measurements =particles2[i].lidar()
        particle_sense_val =0.0
        mean=0.0
        
        #calculating weights for each particle
        for j in range (0,len(U_Buildings )):
            x_dist = particles2[i].x_coord - U_Buildings[j][0]
            y_dist = particles2[i].y_coord - U_Buildings[j][1]
            particle_sense_val = math.sqrt((x_dist*x_dist) + (y_dist*y_dist)) 
            mean = particle_sense_val  #Actual sense value without noise
            prob = scipy.stats.norm.pdf(actual_bot_lidar_measurements[j],mean,
                                        particles2[i].lidar_noise_variance)
            comb_prob_dens = comb_prob_dens*prob
            
        weights.append(comb_prob_dens)
    max_weight_idx = np.argmax(weights) 
    
    

    
    #TEST1
    
    particles3 =[]
    while (len(particles3)<num_particles):
        comb_prob_dens =1.0
        particle_sense_val =0.0
        mean=0.0
        tempo = mobilerobot()
        x_temp = int(np.random.normal((particles2[np.argmax(weights)].x_coord),
                                      20/(c+1)))
        y_temp = int(np.random.normal((particles2[np.argmax(weights)].y_coord),
                                      20/(c+1)))
        orient_temp = ((particles2[np.argmax(weights)].orientation)*
                       (np.random.uniform(low=0.0, high=np.pi)))
        tempo.set_robot_location(x_temp,  y_temp, orient_temp, 0)
        particles3.append(tempo)
            
        
    
    fig = plt.figure(figsize= (10,10))
    plt.xlim(0, 200);
    plt.ylim(0, 200);
    
    
    
        
    for k in range(0, len(particles2)):
        plt.plot(particles3[k].x_coord, particles3[k].y_coord,'ro',markersize=3,markerfacecolor='r',
             markeredgewidth=.01, markeredgecolor='r')
    
    #for k in range(0, len(particles3)):
    #    plt.plot(particles3[k].x_coord, particles3[k].y_coord,'go',markersize=3)
    plt.plot(actualbot.x_coord,actualbot.y_coord,'bo',markersize=10) 
    
    for  h in range(0,len(U_Buildings)):
        mark_x = U_Buildings[h][0]
        mark_y = U_Buildings[h][1]
        plt.plot(mark_x, mark_y, "ko",markersize=10);
    #plt.title('Particles as Red dots, Actual bot as Blue dot, Estimated bot as Green dot')
        
    particles2=[]
    for l in range(0, len(particles3)):
        particles2.append(particles3[l])
        
    Errors =[]
    for g in range(0, len(particles3)):
        Errors.append(error_val(actualbot,particles3[g]))
    
    Err_val = sum(Errors)/num_particles
    print("Errors: ",Err_val)
    
    Err_plot.append(Err_val)
    
  
Iterations=[s for s in range(1,total_iterations+1)]
fig = plt.figure(figsize= (10,10))

plt.plot(Iterations,Err_plot) 
plt.xlabel("Iterations")
plt.ylabel("Error Value")
plt.title("Error Value Vs Iteration")

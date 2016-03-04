# -*- coding: utf-8 -*-
"""
Battery dispatch model. Designed for Cython compilation.

All variables are declared, cimports are used

Use setup.py to compile this file:
python setup.py build_ext --inplace

Versions:
v1: No limitation on the battery power
v2: Battery power limited by the MaxPower variable

Created on Mon Jan 11 20:22:49 2016

@author: Sylvain Quoilin
"""

import numpy as np
cimport numpy as np


def yearly_simulation(np.ndarray[double,ndim=1] PV, np.ndarray[double,ndim=1] load, double timestep, double BatteryCapacity, double BatteryEfficiency, double InverterEfficiency):
    #Initialize variables:
    cdef int Nsteps
    cdef np.ndarray[double,ndim=1] LevelOfCharge, EnergyWithGrid, BatteryGeneration, BatteryConsumption, NetDemandDC
    cdef MaxDischarge, MaxCharge, TotalLoad, TotalFromGrid, SelfConsumption, SelfSufficiencyRate

    Nsteps = len(PV)
    LevelOfCharge = np.zeros(Nsteps)        # kWh
    EnergyWithGrid = np.zeros(Nsteps)       # kWh, egoistical convention (> 0 if buying from grid)
    BatteryGeneration = np.zeros(Nsteps)    # kW 
    BatteryConsumption = np.zeros(Nsteps)   # kW

    if BatteryCapacity > 0:
        NetDemandDC = load/InverterEfficiency - PV
        LevelOfCharge[0] = BatteryCapacity/2
        
        for i in range(1,Nsteps):
            MaxDischarge = LevelOfCharge[i-1]*BatteryEfficiency/timestep
            MaxCharge = (BatteryCapacity - LevelOfCharge[i-1])/timestep
            BatteryGeneration[i] = np.minimum(MaxDischarge,np.maximum(0,NetDemandDC[i]))
            BatteryConsumption[i] = np.minimum(MaxCharge,np.maximum(0,-NetDemandDC[i]))
            LevelOfCharge[i] = LevelOfCharge[i-1] + BatteryConsumption[i] * timestep - BatteryGeneration[i] / BatteryEfficiency * timestep
            EnergyWithGrid[i] = (load[i] - (PV[i] + BatteryGeneration[i] - BatteryConsumption[i])*InverterEfficiency)*timestep

    else:
        EnergyWithGrid = (load - PV*InverterEfficiency)*timestep

    EnergyFromGrid = np.maximum(0,EnergyWithGrid)
    TotalLoad = load.sum()*timestep    
    TotalFromGrid = np.sum(EnergyFromGrid)
    SelfConsumption = TotalLoad - TotalFromGrid   
    SelfSufficiencyRate = float(SelfConsumption)/TotalLoad * 100          # in % 

    return SelfSufficiencyRate
    
    
    
    
def yearly_simulation_v2(np.ndarray[double,ndim=1] PV, np.ndarray[double,ndim=1] load, double timestep, double BatteryCapacity, double BatteryEfficiency, double MaxPower, double InverterEfficiency):
    #Initialize variables:
    cdef int Nsteps
    cdef np.ndarray[double,ndim=1] LevelOfCharge, EnergyWithGrid, BatteryGeneration, BatteryConsumption, NetDemandDC
    cdef MaxDischarge, MaxCharge, TotalLoad, TotalFromGrid, SelfConsumption, SelfSufficiencyRate

    Nsteps = len(PV)
    LevelOfCharge = np.zeros(Nsteps)        # kWh
    EnergyWithGrid = np.zeros(Nsteps)       # kWh, egoistical convention (> 0 if buying from grid)
    BatteryGeneration = np.zeros(Nsteps)    # kW 
    BatteryConsumption = np.zeros(Nsteps)   # kW

    if BatteryCapacity > 0:
        NetDemandDC = load/InverterEfficiency - PV
        LevelOfCharge[0] = BatteryCapacity/2
        
        for i in range(1,Nsteps):
            MaxDischarge = np.minimum(LevelOfCharge[i-1]*BatteryEfficiency/timestep,MaxPower)
            MaxCharge = np.minimum((BatteryCapacity - LevelOfCharge[i-1])/timestep,MaxPower)
            BatteryGeneration[i] = np.minimum(MaxDischarge,np.maximum(0,NetDemandDC[i]))
            BatteryConsumption[i] = np.minimum(MaxCharge,np.maximum(0,-NetDemandDC[i]))
            LevelOfCharge[i] = LevelOfCharge[i-1] + BatteryConsumption[i] * timestep - BatteryGeneration[i] / BatteryEfficiency * timestep
            EnergyWithGrid[i] = (load[i] - (PV[i] + BatteryGeneration[i] - BatteryConsumption[i])*InverterEfficiency)*timestep

    else:
        EnergyWithGrid = (load - PV*InverterEfficiency)*timestep

    EnergyFromGrid = np.maximum(0,EnergyWithGrid)
    TotalLoad = load.sum()*timestep    
    TotalFromGrid = np.sum(EnergyFromGrid)
    SelfConsumption = TotalLoad - TotalFromGrid   
    SelfSufficiencyRate = float(SelfConsumption)/TotalLoad * 100          # in % 

    return SelfSufficiencyRate
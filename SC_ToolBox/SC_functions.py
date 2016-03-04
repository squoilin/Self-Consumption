# -*- coding: utf-8 -*-
"""
Set of common functions required for the Self-Consumption toolbox

Created on Wed Nov 11 22:35:36 2015

@author: Sylvain Quoilin, JRC
"""
from __future__ import division
import numpy as np
import sys



def FinancialAnalysis(E,CountryData,Inv):
    '''
    Calculation of the profits linked to the PV/battery installation, user perspective
       
    :param E: Output of the "EnergyFlows" function: dictionnary of the computed yearly qunatities relative to the PV battery installation. 
    :param CountryData: Dictionary with the financial variables of the considered country   
    :param Inv: Investment data. Defined as a dictionary with the fields 'FixedPVCost','PVCost_kW','FixedBatteryCost','BatteryCost_kWh','PVLifetime','BatteryLifetime','OM'

    :return: List comprising the Profitability Ratio and the system LCOE
    '''
    
    # Defining output dictionnary
    out = {}
    
    # Updating the fixed costs PV or batteries capacities = 0
    if E['CapacityPV'] == 0:
        FixedPVCost = 0
    else:
        FixedPVCost = Inv['FixedPVCost']
    if E['CapacityBattery'] == 0:
        FixedBatteryCost = 0
    else:
        FixedBatteryCost = Inv['FixedBatteryCost']
     
    # Load country Data:
    P_retail = CountryData['P_retail']
    P_support_SC = CountryData['P_support_SC']      # Support to self-consumption, €/MWh
    C_grid_SC = CountryData['C_grid_SC']            # Grid fees for self-consumed electricity, €/MWh
    C_grid_fixed = CountryData['C_grid_fixed']      # Fixed grid tariff per year, €
    C_grid_kW = CountryData['C_grid_kW']            # Fixed cost per installed grid capacity, €/kW
    C_TL_SC = CountryData['C_TL_SC']                # Tax and levies for self-consumed electricity, €/MWh
    P_FtG = CountryData['P_FtG']                    # Purchase price of electricity fed to the grid, €/MWh
    C_grid_FtG = CountryData['C_grid_FtG']          # Grid fees for electricity fed to the grid, €/MWh
    C_TL_FtG = CountryData['C_TL_FtG']              # Tax and levies for electricity fed to the grid, €/MWh
    supportPV_INV = CountryData['support_INV']        # Investment support, % of investment
    supportPV_kW = CountryData['support_kW']          # Investment support proportional to the size, €/kW
    supportBat_INV = 0                          # to be added
    supportBat_kW = 0                           # to be added
    i = CountryData['WACC']                         # Discount rate, -
    net_metering = CountryData['net_metering']      # Boolean variable for the net metering scheme    
    
    # Battery investment with one reimplacement after the battery lifetime (10 years)
    NPV_Battery_reinvestment = (FixedBatteryCost + Inv['BatteryCost_kWh'] * E['CapacityBattery']) / (1+i)**Inv['BatteryLifetime']
    BatteryInvestment = (FixedBatteryCost + Inv['BatteryCost_kWh'] * E['CapacityBattery']) + NPV_Battery_reinvestment
    
    # PV investment:
    PVInvestment = FixedPVCost + Inv['PVCost_kW'] * E['CapacityPV']
    
    # Investment costs:
    NetSystemCost = PVInvestment * (1 - supportPV_INV) - supportPV_kW * E['CapacityPV']  \
                    + BatteryInvestment * (1 - supportBat_INV) - supportBat_kW * E['CapacityBattery']
    CRF = i * (1+i)**Inv['PVLifetime']/((1+i)**Inv['PVLifetime']-1)  # Capital Recovery Factor, %
    AnnualInvestment = NetSystemCost * CRF + Inv['OM'] * (BatteryInvestment + PVInvestment)
    
    # Total investment without the subsidies (O&M could also possibly be removed...):
    ReferenceAnnualInvestment = (BatteryInvestment + PVInvestment) * (CRF  + Inv['OM'])
    ReferenceAnnualInvestment = np.maximum(1e-7,ReferenceAnnualInvestment)       # avoids division by zero
    
    # Annual costs:
    AnnualCost = C_grid_fixed + C_grid_kW * E['CapacityPV']
    
    if net_metering:
        # Revenues:
        Income_FtG = np.maximum(0,E['ACGeneration']-E['Load']) * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = (P_support_SC + P_retail - C_grid_SC - C_TL_SC) * np.minimum(E['ACGeneration'],E['Load'])/1000  # the retail price on the self-consumed part is included here since it can be considered as a support to SC    
        # Cost of electricity bought to the grid:
        Cost_BtG = P_retail * np.maximum(E['Load']-E['ACGeneration'],0)/1000        
    else:
        # Revenues:
        Income_FtG = E['ToGrid'] * (P_FtG - C_grid_FtG - C_TL_FtG)/1000
        Income_SC = (P_support_SC + P_retail - C_grid_SC - C_TL_SC) * E['SC']/1000  # the retail price on the self-consumed part is included here since it can be considered as a support to SC
        # Cost of electricity bought to the grid:
        Cost_BtG = P_retail * E['FromGrid']/1000

    Profit =  Income_FtG + Income_SC - AnnualInvestment - AnnualCost
    out['PR'] = Profit/ReferenceAnnualInvestment*100
    
    # Calculating LCOE as if the grid was a generator
    out['LCOE'] = (AnnualInvestment + AnnualCost - Income_FtG - (P_support_SC - C_grid_SC - C_TL_SC) * E['SC']/1000 + Cost_BtG)/E['Load']
    #LCOE = -(-AnnualInvestment - AnnualCost + Income_FtG + Income_SC)
    
    # LCOE of storage only:
    if E['FromBattery'] > 1:
        out['LCOE_stor'] = BatteryInvestment * ( CRF + Inv['OM']) / E['FromBattery']
    else:
        out['LCOE_stor'] = np.nan

    return out





def EnergyFlows(r_PV,r_bat,Load,eta_inv,eta_bat,CapacityFactorPV,a):
    '''
    The EnergyFlows function calculates the different yearly energy quantities 
    in each component of the PV/battery/household/grid system
    
    :param r_PV: Ratio of the PV DC production (i.e. before inverter) to the total household demand
    :param r_bat: Battery size in kWh divided by total household demand in MWh
    :param Load: Total household demand, kWh
    :param eta_inv: Inverter efficiency, -
    :param eta_bat: Battery round-trip efficiency, -
    :param CapacityFactorPV: Capacity factor of a typical PV installation in the considered area, kWh/kWp
    
    :return: Dictionary with the main energy flows
    '''
    E = {}
    
    # Total demand:
    E['Load'] = Load
    
    # PV generation (DC):
    E['PV_DC'] = r_PV * Load
    
    # PV installed capacity (AC):
    E['CapacityPV'] = E['PV_DC']*eta_inv/CapacityFactorPV
    
    # Battery size (only the accessible capacity):
    E['CapacityBattery'] = r_bat * Load / 1000
    
    # Self-Sufficiency rate:
    E['SSR'] = fSSR(r_PV,r_bat,a)
    
    # Self-Sufficiency Rate without battery:
    E['SSR_0'] = fSSR(r_PV,0,a)
    
    # Self-Consumption Rate (defined as the AC self-consumption divided by the DC PV production!):
    if r_PV > 0:
        E['SCR'] = E['SSR'] / r_PV
    else:
        E['SCR'] = 100

    # Self-Consumption on the DC side, no battery
    SC_DC_0 = E['SSR_0'] * Load / eta_inv / 100

    # Self-Consumption on the AC and DC sides, with battery:
    E['SC'] = E['SSR'] * Load / 100
    SC_DC = E['SC']/eta_inv
    
    # The energy coming out of the battery is the self-consumption minus the self-consumption in the case without battery
    E['FromBattery'] = SC_DC - SC_DC_0
    
    # The energy flowing to the battery is calculated with the battery round-trip efficiency:
    E['ToBattery'] = E['FromBattery']/eta_bat
    
    # The amount of electricity sold to the grid is what remains from the PV production after removing the self-consumped flows:
    E['ToGrid'] = eta_inv*(E['PV_DC'] - SC_DC_0 - E['ToBattery'])
    
    # The amount of electricity bougth to the grid is the load minus self-consumption
    E['FromGrid'] = Load - E['SC']
    
    # Total amount of AC generation (i.e after the inverter):
    E['ACGeneration'] = E['ToGrid'] + E['SC']

    return E


def fSSR(r_PV,r_bat, a):
    '''
    SSR as a function of the PV and battery relative sizes. This function is declined
    in two versions: the V1 counts 10 parameters and V2 counts 16 parameters
    :param r_PV: Yearly PV generation divided by yearly demand (kWh/kWh)
    :param r_bat: Battery capacity divided by yearly demand (kWh,MWh)
    :param a: List of 10/16 empirical coefficients
    '''
    
    if len(a) == 10:   # First version
        if r_PV == 0:
            SSR = 0
        elif r_PV < 1:
            SSR = r_PV * ((a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat) + (a[7] * np.tanh(a[8] * (1 - r_PV))))
        else:
            SSR = ((a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat) +  ((a[4] + a[9]* r_bat) * np.tanh(a[5]*(r_PV-1))+ a[6]* (r_PV-1)))
        
        return np.minimum(100,np.maximum(0,SSR))
        
    elif len(a) == 16:          # Second version
        # Weigthing factor to impose an accuracte, fitted curve at r_bat = 0:
        max_W = 0.4
        W = np.minimum(max_W,np.maximum(0,r_bat))
        
        
        if r_PV == 0:
            SSR = 0
        elif r_PV < 1:
            SSR = W/max_W * \
                  (a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat + \
                  (a[12] * np.tanh(a[13] * (1 - r_PV)))) * r_PV + \
                  (max_W - W)/max_W * \
                  np.maximum(r_PV, (a[4] * np.tanh(a[5]*r_PV)+ a[6]* r_PV + a[7]* r_PV**0.5) \
                  * ( 1 + a[14]*np.tanh(r_bat))+a[15]*np.tanh(r_bat))
        else:
            SSR = W/max_W * \
                  (a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat + \
                  (a[8] + a[9]* r_bat) * np.tanh(a[10]*(r_PV-1))+ a[11]* (r_PV-1)) + \
                  (max_W - W)/max_W * \
                  np.maximum(r_PV, (a[4] * np.tanh(a[5]*r_PV)+ a[6]* r_PV + a[7]* r_PV**0.5) \
                  * ( 1 + a[14]*np.tanh(r_bat))+a[15]*np.tanh(r_bat))   
        return np.minimum(100,np.maximum(0,SSR))   
    else:
        sys.exit('The number of coefficients of the fSSR function should be 10 or 16')


def SelfConsumption(PV,Demand):
    ''' 
    Simplistic function computing the self-consumption and self-sufficiency rates
    from PV and demand vectors
    '''
    aa = np.minimum(PV,Demand)
    SCR = aa.sum()/PV.sum()
    SSR = aa.sum()/Demand.sum()
    return SCR,SSR
    
def scale_vector(vec_in,N,silent=False):
    ''' 
    Function that scales a numpy vector of Pandas Series to the desired length
    
    :param vec_in: Input vector
    :param N: Length of the output vector
    :param silent: Set to True to avoid verbosity
    '''    
    
    N_in = len(vec_in)
    if type(N) != int:
        N = int(N) 
        if not silent:
            print 'Converting Argument N to int: ' + str(N)
    if N > N_in:
        if np.mod(N,N_in)==0:
            if not silent:
                print 'Target size is a multiple of input vector size. Repeating values'
            vec_out = np.repeat(vec_in,N/N_in)
        else:
            if not silent:
                print 'Target size is larger but not a multiple of input vector size. Interpolating'
            vec_out = np.interp(np.linspace(start=0,stop=N_in,num=N),range(N_in),vec_in)
    elif N == N_in:
        print 'Target size is iqual to input vector size. Not doing anything'
        vec_out = vec_in
    else:
        if np.mod(N_in,N)==0:
            if not silent:
                print 'Target size is entire divisor of the input vector size. Averaging'
            vec_out = np.zeros(N)
            for i in range(N):
                vec_out[i] = np.mean(vec_in[i*N_in/N:(i+1)*N_in/N])
        else:
            if not silent:
                print 'Target size is lower but not a divisor of the input vector size. Interpolating'
            vec_out = np.interp(np.linspace(start=0,stop=N_in,num=N),range(N_in),vec_in)
    return vec_out    


def battery_simulation(PV,load,param,print_analysis=False,output_timeseries=True):
    ''' 
    Battery dispatch algorithm.
    The dispatch of the storage capacity is performed in such a way to maximize self-consumption:  
    the battery is charged when the PV power is higher than the load and as long as it is not fully charged. 
    It is discharged as soon as the PV power is lower than the load and as long as it is not fully discharged.

    :param PV: Vector of PV generation, in kW DC (i.e. before the inverter)
    :param load: Vector of household consumption, kW
    :param param: Dictionnary with the simulation parameters: 
                    timestep: Simulation time step (in hours)
                    BatteryCapacity: Available battery capacity (i.e. only the the available DOD), kWh
                    BatteryEfficiency: Battery round-trip efficiency, -
                    InverterEfficiency: Inverter efficiency, -
                    MaxPower: Maximum battery charging or discharging powers (assumed to be equal), kW
    
    :return: Self-Sufficiency Rate (SSR), in %
    
    '''

    #Initialize variables:
    Nsteps = len(PV)
    timestep = param['timestep']
    BatteryCapacity = param['BatteryCapacity']
    BatteryEfficiency = param['BatteryEfficiency']
    InverterEfficiency = param['InverterEfficiency']
    LevelOfCharge = np.zeros(Nsteps)        # kWh
    EnergyWithGrid = np.zeros(Nsteps)       # kWh, selfish convention (> 0 if buying from grid)
    BatteryGeneration = np.zeros(Nsteps)    # kW 
    BatteryConsumption = np.zeros(Nsteps)   # kW
    if 'MaxPower' in param:
        MaxPower = param['MaxPower']
    else:
        MaxPower = 1E15    
    
    if BatteryCapacity > 0:
        LevelOfCharge[0] = BatteryCapacity/2
        NetDemandDC = load/InverterEfficiency - PV
    
        for i in range(1,Nsteps):
            MaxDischarge = np.minimum(MaxPower,LevelOfCharge[i-1]*BatteryEfficiency/timestep)
            MaxCharge = np.minimum(MaxPower,(BatteryCapacity - LevelOfCharge[i-1])/timestep)
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
    SelfSufficiencyRate = float(SelfConsumption)/TotalLoad * 100 # in % 

    if print_analysis:
        EnergyToGrid = np.maximum(0,-EnergyWithGrid)
        TotalFromGrid = np.sum(EnergyFromGrid)
        TotalToGrid = np.sum(EnergyToGrid)
        TotalPV = PV.sum()*timestep
        TotalBatteryGeneration = np.sum(BatteryGeneration)*timestep
        TotalBatteryConsumption = np.sum(BatteryConsumption)*timestep
        BatteryLosses = TotalBatteryConsumption - TotalBatteryGeneration
        InverterLosses = (TotalPV - BatteryLosses)*(1-InverterEfficiency)
        SelfConsumptionRate = SelfConsumption/TotalPV*100             # in % 
        AverageDepth = np.sum(BatteryGeneration*timestep)/(365 * BatteryCapacity)
        Nfullcycles = 365*AverageDepth
        residue = TotalPV + TotalFromGrid - TotalToGrid - BatteryLosses - InverterLosses - TotalLoad
    
        print 'Total yearly consumption: ' + str(TotalLoad) + ' kWh'
        print 'Total PV production: ' + str(TotalPV) + ' kWh'
        print 'Self Consumption: ' + str(SelfConsumption) + ' kWh'
        print 'Total fed to the grid: ' + str(TotalToGrid) + ' kWh'
        print 'Total bought from the grid: ' + str(TotalFromGrid) + ' kWh'
        print 'Self consumption rate (SCR): '  + str(SelfConsumptionRate) + '%'
        print 'Self sufficiency rate (SSR): ' + str(SelfSufficiencyRate) + '%'
        print 'Amount of energy provided by the battery: ' + str(TotalBatteryGeneration) + ' kWh'
        print 'Average Charging/Discharging depth: ' + str(AverageDepth)
        print 'Number of equivalent full cycles per year: ' + str(Nfullcycles)
        print 'Total battery losses: ' + str(BatteryLosses) + ' kWh'
        print 'Total inverter losses: ' + str(InverterLosses) + ' kWh'
        print 'Residue (check) :' + str(residue) + 'kWh \n'
    
    if output_timeseries:
        return [BatteryGeneration,BatteryConsumption,LevelOfCharge,EnergyWithGrid]
    else:
        return SelfSufficiencyRate
import pyroomacoustics as pra
from pyroomacoustics.directivities import (
    MeasuredDirectivityFile, Rotation3D)
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt
from scipy.signal import fftconvolve
import csv

csv_filename = "room_metadata.csv"
csv_fields = ["filename", "room", "azimuth", "max_order", "distance", "absorption","rt60_early", "rt60_late", "rt60_merged"]

with open(csv_filename, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()

small = {
    "name": "small",
    "dimensions": [3, 4, 3],
    "mic":[1.5, 2, 1.5],
    "dist":[1]
}    

medium = {
    "name": "medium",
    "dimensions": [8, 6, 3],
    "mic":[4, 3, 1.5],
    "dist":[1, 2]
}    

big = {
    "name": "big",
    "dimensions": [10, 15, 3],
    "mic":[5, 7.5, 1.5],
    "dist":[1, 2, 4]
}    

rooms = [small, medium, big]    

az_values = [0, 30, 60, 90, 120, 150, 180]
max_order_val = [0, 1, 2, 3]
materials = [0.2, 0.8]

fs = 44100
fs_wav, signal = wavfile.read("speech.wav")
sofa_path = "D2_44K_16bit_256tap_FIR_SOFA.sofa"

sofa_data = MeasuredDirectivityFile(sofa_path,fs=fs)
orient = Rotation3D([0, 0, 0])
directivities = [
    sofa_data.get_mic_directivity(0, orient),
    sofa_data.get_mic_directivity(1, orient)]

def source_location (az, dist, mic):
    az_rad = np.deg2rad(az)
    source_x = mic[0] + dist * np.cos(az_rad)
    source_y = mic[1] + dist * np.sin(az_rad)
    source_z = mic[2] 
    source_loc = [source_x, source_y, source_z]
    return source_loc

def merge_rirs(rir_ism, rir_ray, fs, mix_samp):


    fade_len = int(0.01 * fs) 
    
    length = max(len(rir_ism), len(rir_ray))
    rir_ism = np.pad(rir_ism, (0, length - len(rir_ism)))
    rir_ray = np.pad(rir_ray, (0, length - len(rir_ray)))
    
    energy_ism = np.sum(rir_ism[mix_samp-fade_len : mix_samp]**2)
    energy_ray = np.sum(rir_ray[mix_samp-fade_len : mix_samp]**2)
    gain = np.sqrt(energy_ism / (energy_ray + 1e-10))
    rir_ray *= gain


    fade_in = np.linspace(0, 1, fade_len)
    fade_out = 1 - fade_in
    combined = np.zeros(length)
    
    combined[:mix_samp-fade_len] = rir_ism[:mix_samp-fade_len]


    combined[mix_samp-fade_len : mix_samp] = (
        rir_ism[mix_samp-fade_len : mix_samp] * fade_out + 
        rir_ray[mix_samp-fade_len : mix_samp] * fade_in
    )
    combined[mix_samp:] = rir_ray[mix_samp:]
    
    return combined

def hybrid_room(room, room_dim, az, signal, max_ord, material, mic, dist):

    mic_locs = np.array([
        [mic[0], mic[1] , mic[2]], 
        [mic[0], mic[1], mic[2]],  
    ]).T

    source_loc = source_location(az,dist,mic)
    materials = pra.Material(material)

    ##### ISM
    room_early = pra.ShoeBox(room_dim, fs=fs, materials=materials, max_order=max_ord)
    room_early.add_microphone_array(mic_locs, directivity=directivities)
    room_early.add_source(source_loc, signal=signal)
    room_early.simulate()
    rir_early_L = room_early.rir[0][0]
    rir_early_R = room_early.rir[1][0]
    rt60_early = pra.measure_rt60(rir_early_L,fs=fs)
    
    #### Ray Tracing
    room_late = pra.ShoeBox(room_dim, fs=fs, materials=materials, max_order=max_ord, ray_tracing = True, air_absorption=True)
    room_late.set_ray_tracing(
        receiver_radius=0.3,   
        n_rays=10000,           
        energy_thres=1e-7,      
    )
    room_late.add_microphone_array(mic_locs)
    room_late.add_source(source_loc, signal=signal)
    room_late.simulate()
    rir_late = room_late.rir[0][0]
    rt60_late = pra.measure_rt60(rir_late, fs=fs)

    ###### Mix RIRs ###################################
    arrival_L = int(np.where(np.abs(rir_early_L) > (0.1 * np.max(np.abs(rir_early_L))))[0][0])
    arrival_R = int(np.where(np.abs(rir_early_R) > (0.1 * np.max(np.abs(rir_early_R))))[0][0])
    arrival_late = int(np.where(np.abs(rir_late) > (0.1 * np.max(np.abs(rir_late))))[0][0])
    
    target_arrival = min(arrival_L, arrival_R)
    shift = target_arrival - arrival_late 
    
    if shift >= 0:
        rir_late_aligned = np.concatenate([np.zeros(shift), rir_late])
    else:
        rir_late_aligned = rir_late[-shift:]  
    
    # Transition: 30-50ms ΜΕΤΑ το arrival 
    mix_samp = target_arrival + int(0.030 * fs)
    merged_L = merge_rirs(rir_early_L, rir_late_aligned, fs, mix_samp=mix_samp)
    merged_R = merge_rirs(rir_early_R, rir_late_aligned, fs, mix_samp=mix_samp)
    rt60_merged_L = pra.measure_rt60(merged_L,fs=fs)

    signal = signal.astype(np.float32) / 32768.0
    left_channel = fftconvolve(signal, merged_L, mode='full')
    right_channel = fftconvolve(signal, merged_R, mode='full')
    stereo_signal = np.vstack((left_channel, right_channel)).T
    
    max_val = np.max(np.abs(stereo_signal))
    if max_val > 0:
        stereo_signal = stereo_signal / max_val
    stereo_signal_int = (stereo_signal * 32767).astype(np.int16)

    filename = f"binaural__{room}_max_order{max_ord}_{az}_{material}_{dist}.wav"
    wavfile.write(filename, fs, stereo_signal_int)

    # Append this file's metadata as a new row
    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writerow({
            "filename": filename,
            "room": room,
            "azimuth": az,
            "max_order": max_ord,
            "distance": dist,
            "absorption": material,
            "rt60_early": rt60_early,
            "rt60_late":rt60_late,
            "rt60_merged": rt60_merged_L
        })
    


for room in rooms:
    for az in az_values:
        for max_ord in max_order_val:
            for material in materials: 
                for dist in room['dist']:
                    try:
                        hybrid_room(room["name"], room["dimensions"], az, signal, max_ord, material, room["mic"], dist)
                    except Exception as e:
                        print(f"FAILED: {room['name']} az={az} ord={max_ord} mat={material} dist={dist} — {e}")
    print(f"{room['name']} done")
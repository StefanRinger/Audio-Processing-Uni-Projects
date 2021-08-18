from matplotlib import transforms
import matplotlib.pyplot as plt
import time
import sys
import numpy as np
import soundfile as sf
from librosa.core import resample
from librosa.core import istft, stft
import IPython.display as ipd
from ipywidgets import *



def evaluate_dsb_interferer(dsb, dsb_name):
    sf.write(str(dsb_name)+'_source_noise_inft_unp.wav', dsb.env.noised_mix_mics[0,...], int(dsb.env.f_max*2))
    #dsb0.compute_weights(compute_steering_vector(dsb0.env.s, dsb0.env.ULA_pos, dsb0.env.ULA_direction, dsb0.env.freqs, dsb0.env.d, dsb0.env.c, dsb0.env.N))
    stft_matrix = np.asarray([ stft(sig,n_fft=int(dsb.env.os_fft_bins*2-2)) for sig in list(dsb.env.noised_mix_mics)])
    dsb_source_noise_pr = dsb.apply_weights(stft_matrix)
    print(dsb_source_noise_pr.shape)
    sf.write(str(dsb_name)+'_source_noise_intf_pro.wav', istft(dsb_source_noise_pr), int(dsb.env.f_max*2))
    sf.write(str(dsb_name)+'_source_noise_intf_av.wav', istft(np.mean(stft_matrix,axis=0)), int(dsb.env.f_max*2))
    fig, ax = plt.subplots(3, 1, sharey=True)
    fig.set_size_inches((10,15))
    max_val = 20*np.log(np.max(np.abs(stft_matrix)))
    min_val = 20*np.log(np.min(np.abs(dsb_source_noise_pr)))
    ax[0].set_title('Unprocessed noisy signal at microphone 1')
    ax[0].pcolormesh(20*np.log(np.abs(stft_matrix[0])), vmin=min_val, vmax=max_val)
    ax[1].set_title('DSB')
    ax[1].pcolormesh(20*np.log(np.abs(dsb_source_noise_pr)), vmin=min_val, vmax=max_val)
    ax[2].set_title('Averaged over Microphones')
    ax[2].pcolormesh(20*np.log(np.abs(np.mean(stft_matrix,axis=0))), vmin=min_val, vmax=max_val)
    for axes in ax:
        plt.sca(axes)
        plt.yticks([dsb.env.os_fft_bins* a/8 for a in np.arange(0,9)],np.arange(0,dsb.env.f_max+1,1000))
    plt.sca(ax[1])
    plt.ylabel('Frequency [Hz]')
    plt.sca(ax[2])
    plt.xlabel('Time Frames')
    fig.subplots_adjust(hspace=0.15)
    plt.show()
    print('Unprocessed noisy signal at microphone 1')
    display(ipd.Audio(str(dsb_name)+'_source_noise_inft_unp.wav'))
    print('DSB')
    display(ipd.Audio(str(dsb_name)+'_source_noise_intf_pro.wav'))
    print('Averaged Microphone Signals')
    display(ipd.Audio(str(dsb_name)+'_source_noise_intf_av.wav'))
    fig, ax = plt.subplots(3, 1, sharey=True)
    fig.set_size_inches((10,15))
    ax[0].set_title('Unprocessed noisy signal at microphone 1')
    ax[0].plot(istft(stft_matrix[0]))
    ax[1].set_title('DSB')
    ax[1].plot(istft(dsb_source_noise_pr))
    ax[2].set_title('Averaged over Microphones')
    ax[2].plot(istft(np.mean(stft_matrix,axis=0)))

    plt.sca(ax[1])
    plt.ylabel('Frequency [Hz]')
    plt.sca(ax[2])
    plt.xlabel('Samples')
    fig.subplots_adjust(hspace=0.15)
    plt.show()





def evaluate_dsb(dsb, dsb_name, interferer=False):
    sf.write(str(dsb_name)+'_source_noise_unp.wav', dsb.env.noised_s_mics[0,...], int(dsb.env.f_max*2))
    #dsb0.compute_weights(compute_steering_vector(dsb0.env.s, dsb0.env.ULA_pos, dsb0.env.ULA_direction, dsb0.env.freqs, dsb0.env.d, dsb0.env.c, dsb0.env.N))
    stft_matrix = np.asarray([ stft(sig,n_fft=int(dsb.env.os_fft_bins*2-2)) for sig in list(dsb.env.noised_s_mics)])
    if interferer:
        return evaluate_dsb_interferer(dsb, dsb_name)

    dsb_source_noise_pr = dsb.apply_weights(stft_matrix)
    print(dsb_source_noise_pr.shape)
    sf.write(str(dsb_name)+'_source_noise_pro.wav', istft(dsb_source_noise_pr), int(dsb.env.f_max*2))
    sf.write(str(dsb_name)+'_source_noise_av.wav', istft(np.mean(stft_matrix,axis=0)), int(dsb.env.f_max*2))
    fig, ax = plt.subplots(3, 1, sharey=True)
    fig.set_size_inches((10,15))
    max_val = 20*np.log(np.max(np.abs(stft_matrix)))
    min_val = 20*np.log(np.min(np.abs(dsb_source_noise_pr)))
    ax[0].set_title('Unprocessed noisy signal at microphone 1')
    ax[0].pcolormesh(20*np.log(np.abs(stft_matrix[0])), vmin=min_val, vmax=max_val)
    ax[1].set_title('DSB')
    ax[1].pcolormesh(20*np.log(np.abs(dsb_source_noise_pr)), vmin=min_val, vmax=max_val)
    ax[2].set_title('Averaged over Microphones')
    ax[2].pcolormesh(20*np.log(np.abs(np.mean(stft_matrix,axis=0))), vmin=min_val, vmax=max_val)
    for axes in ax:
        plt.sca(axes)
        plt.yticks([dsb.env.os_fft_bins* a/8 for a in np.arange(0,9)],np.arange(0,dsb.env.f_max+1,1000))
    plt.sca(ax[1])
    plt.ylabel('Frequency [Hz]')
    plt.sca(ax[2])
    plt.xlabel('Time Frames')
    fig.subplots_adjust(hspace=0.15)
    plt.show()
    print('Unprocessed noisy signal at microphone 1')
    display(ipd.Audio(str(dsb_name)+'_source_noise_unp.wav'))
    print('DSB')
    display(ipd.Audio(str(dsb_name)+'_source_noise_pro.wav'))
    print('Averaged Microphone Signals')
    display(ipd.Audio(str(dsb_name)+'_source_noise_av.wav'))
    fig, ax = plt.subplots(3, 1, sharey=True)
    fig.set_size_inches((10,15))
    ax[0].set_title('Unprocessed noisy signal at microphone 1')
    ax[0].plot(istft(stft_matrix[0]))
    ax[1].set_title('DSB')
    ax[1].plot(istft(dsb_source_noise_pr))
    ax[2].set_title('Averaged over Microphones')
    ax[2].plot(istft(np.mean(stft_matrix,axis=0)))
    plt.sca(ax[1])
    plt.ylabel('Frequency [Hz]')
    plt.sca(ax[2])
    plt.xlabel('Samples')
    fig.subplots_adjust(hspace=0.15)
    plt.show()



def compute_masks(desired, undesired, env):
    desired_spec= np.real(np.abs(np.asarray([ stft(sig,n_fft=int(env.os_fft_bins*2-2)) for sig in list(desired)])))
    undesired_spec= np.real(np.abs(np.asarray([ stft(sig,n_fft=int(env.os_fft_bins*2-2)) for sig in list(undesired)])))
    des_mask = np.real(np.zeros_like(np.abs(desired_spec)))
    unddes_mask = np.real(np.ones_like(np.abs(undesired_spec)))
    des_mask[np.abs(desired_spec) > np.abs(undesired_spec)] = 1
    unddes_mask -= des_mask
    unddes_mask = np.mean(unddes_mask, axis = 0)
    unddes_mask[unddes_mask > 0.5] = 1
    unddes_mask[unddes_mask < 1 ] = 0
    return unddes_mask

def compute_power_pattern_for_freq(beamformer, freq, env, compute_doa, compute_tdoa, compute_tdoa_list, compute_steering_for_freq, compute_steering_vector,num_points = 100):
    doa_list = create_sampling_points(num_points)
    theta_list = [compute_doa(point, env.ULA_center, env.ULA_direction) for point in doa_list ]
    #theta_list = [compute_theta(doa, env.ULA_direction) for doa in doa_list]
    tdoa_list = [compute_tdoa(theta, env.d, env.c) for theta in theta_list]
    freq_index = int(round((env.os_fft_bins-1)*freq/env.f_max))
    pos_steerings = []
    for tdoa in tdoa_list:
        pos_steerings.append(compute_steering_for_freq(env.freqs[freq_index][None,...], np.array([tdoa*a for a in range(env.N)])))

    return 10*np.log(np.abs(np.square(np.dot(np.asarray(pos_steerings),np.conjugate(beamformer.weights[freq_index].T)))))

def create_sampling_points(num_points):
    # sample the unit circle with num points
    unit_circle_points = [np.exp(2*np.pi/num_points*point*1j) for point in range(num_points)]
    # transform those into doas
    doa_list = [np.asarray([np.real(a),np.imag(a)]) for a in unit_circle_points]
    return doa_list

def assign_signals_to_environment(env,compute_steering_vector,  intf_path = './audio/SA1.WAV', source_path = './audio/SX141.WAV'):
    interferer, fs_intf = sf.read(intf_path)
    interferer-=np.mean(interferer)
    interferer = 0.8*resample(interferer,fs_intf, int(round(env.f_max*2)))
    source, fs_s = sf.read(source_path)
    source = source -np.mean(source)
    source = resample(source,fs_s, int(round(env.f_max*2)))
    #source = source*np.sqrt(np.sum(np.square(interferer)/len(interferer))/(np.sum(np.square(source)/len(source))))
    source *= np.sqrt(np.mean(np.square(interferer))/np.mean(np.square(source)))
    #plt.figure()
    #plt.plot(source)
    #plt.plot(interferer)
    #plt.show()
    def get_mic_signals(signal, env, source = True):
        if source:
            steering_vec = np.asarray(compute_steering_vector(env.s, env.ULA_center, env.ULA_direction, env.freqs,env.d, env.c, env.N)).T
        else:
            steering_vec = np.asarray(compute_steering_vector(env.intf, env.ULA_center, env.ULA_direction, env.freqs, env.d, env.c, env.N)).T
        mic_signals = []

        for mic in range(env.N):
            mic_signals.append(istft(stft(signal,n_fft=int(env.os_fft_bins*2-2))*np.expand_dims(steering_vec[mic,:],-1)))
        return np.asarray(mic_signals)
    def add_noise_to_mics(mic_signals, env):
        sign_std = np.sqrt(np.sum(np.square(mic_signals[0,...]))/len(mic_signals[0,...]))
        wgn_std = sign_std/10**(env.SNR/20)
        noised_mics = mic_signals + np.random.normal(0,wgn_std,mic_signals.shape)
        return noised_mics
    env.source_mics = get_mic_signals(source, env, source = True)
    env.intf_mics = get_mic_signals(interferer, env, source = False)
    s_len = env.source_mics.shape[-1]
    intf_len = env.intf_mics.shape[-1]
    if s_len > intf_len:
        env.intf_mics = np.pad(env.intf_mics, ((0,0),(0,s_len - intf_len)), 'constant')
    else:
        env.source_mics = np.pad(env.source_mics, ((0,0),(0,intf_len - s_len)), 'constant')
    env.mix = env.source_mics + env.intf_mics
    # I rewrote that to only have one noise class ...
    #env.noised_s_mics = add_noise_to_mics(env.source_mics, env)



    #env.noised_intf_mics = add_noise_to_mics(env.intf_mics, env)
    #env.noised_mix_mics = add_noise_to_mics(env.mix, env)
    ############new
    env.noised_s_mics = add_noise_to_mics(env.source_mics, env)
    env.noise = env.noised_s_mics - env.source_mics
    env.noised_mix_mics = env.mix + env.noise
def visualize_dsb_power_pattern(dsb, compute_doa, compute_tdoa, compute_tdoa_list, compute_steering_for_freq, compute_steering_vector):
    plt.figure()
    env = dsb.env
    num_points = env.num_points
    fs = int(env.f_max*2)
    K =  int(env.os_fft_bins*2-2)
    r = compute_power_pattern_for_freq(dsb, num_points, env,compute_doa, compute_tdoa, compute_tdoa_list, compute_steering_for_freq, compute_steering_vector, num_points = num_points)
    theta = np.asarray([point*2*np.pi/num_points for point in range(num_points)])
    #rot = transforms.Affine2D().rotate_deg(90)
    whole = np.squeeze(np.asarray([compute_power_pattern_for_freq(dsb,freq, env, compute_doa, compute_tdoa, compute_tdoa_list, compute_steering_for_freq, compute_steering_vector,num_points = num_points) for freq in np.arange(0,env.f_max+1, env.f_max/(env.os_fft_bins-1))]))

    plt.title('Beam Pattern in dB')
    plt.xticks(range(0,num_points + 1,int(num_points/8)),np.arange(0,361,360/8))
    plt.yticks(env.freqs,np.arange(0,env.f_max+1,1000))
    plt.xlabel('Angle of Arrival')
    plt.ylabel('Frequency')
    plt.pcolormesh(whole)
    plt.colorbar()
    plt.show()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    line,  = ax.plot(theta, np.abs(r))
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_rticks([-6,-3, 0])
    ax.set_rlabel_position(1)
    ax.set_theta_offset(0)
    ax.grid(True)
    ax.set_title("Beam Pattern in dB", va='bottom')
    fig.show()

    def update(Hz = np.arange(0,env.f_max+1,250)):
        line.set_ydata(compute_power_pattern_for_freq(dsb, Hz, env,compute_doa, compute_tdoa, compute_tdoa_list, compute_steering_for_freq, compute_steering_vector, num_points = num_points))
        fig.canvas.draw()

    def sweep(b):
        for Hz in np.arange(0,env.f_max+1,100):
            line.set_ydata(compute_power_pattern_for_freq(dsb, Hz, env, compute_doa, compute_tdoa, compute_tdoa_list, compute_steering_for_freq, compute_steering_vector,num_points = num_points))
            time.sleep(0.1)
            print("\rFrequency: {} Hz ".format(Hz), end="")
            sys.stdout.flush()
            fig.canvas.draw()

    button = widgets.Button(description="Beam-Sweep")
    display(button)
    button.on_click(sweep)
    interact(update);




def evaluate_beamformers(mvdr0,mvdr_name,dsb0, dsb_name):
    sf.write(str(mvdr_name)+ '_source_noise_inft_unp.wav', mvdr0.env.noised_mix_mics[0,...], int(mvdr0.env.f_max*2))
    #dsb0.compute_weights(compute_steering_vector(dsb0.env.s, dsb0.env.ULA_pos, dsb0.env.ULA_direction, dsb0.env.freqs, dsb0.env.d, dsb0.env.c, dsb0.env.N))
    stft_matrix = np.asarray([ stft(sig,n_fft=int(mvdr0.env.os_fft_bins*2-2)) for sig in list(mvdr0.env.noised_mix_mics)])
    mvdr_source_intf_noise_pr = mvdr0.apply_weights(stft_matrix)
    sf.write(str(mvdr_name)+ '_source_noise_intf_pro.wav', istft(mvdr_source_intf_noise_pr), int(mvdr0.env.f_max*2))
    sf.write(str(mvdr_name)+ '_source_noise_intf_av.wav', istft(np.mean(stft_matrix,axis=0)), int(mvdr0.env.f_max*2))
    fig, ax = plt.subplots(3, 1, sharey=True)
    dsb_source_intf_noise_pr = dsb0.apply_weights(stft_matrix)
    sf.write(str(dsb_name)+ '_source_noise_intf_pro.wav', istft(dsb_source_intf_noise_pr), int(dsb0.env.f_max*2))
    fig.set_size_inches((10,15))
    max_val = 20*np.log(np.max(np.abs(stft_matrix)))
    min_val = 20*np.log(np.min(np.abs(mvdr_source_intf_noise_pr)))
    ax[0].set_title('Unprocessed noisy signal at microphone 1')
    ax[0].pcolormesh(20*np.log(np.abs(stft_matrix[0])), vmin=min_val, vmax=max_val)
    ax[1].set_title(mvdr_name)
    ax[1].pcolormesh(20*np.log(np.abs(mvdr_source_intf_noise_pr)), vmin=min_val, vmax=max_val)
    ax[2].set_title(dsb_name)
    ax[2].pcolormesh(20*np.log(np.abs(dsb_source_intf_noise_pr)), vmin=min_val, vmax=max_val)
    for axes in ax:
        plt.sca(axes)
        plt.yticks([mvdr0.env.os_fft_bins* a/8 for a in np.arange(0,9)],np.arange(0,mvdr0.env.f_max+1,1000))
    plt.sca(ax[1])
    plt.ylabel('Frequency [Hz]')
    plt.sca(ax[2])
    plt.xlabel('Time Frames')
    fig.subplots_adjust(hspace=0.15)
    plt.show()
    print('Unprocessed noisy signal at microphone 1')
    display(ipd.Audio(str(mvdr_name)+ '_source_noise_inft_unp.wav'))
    print(mvdr_name)
    display(ipd.Audio(str(mvdr_name)+ '_source_noise_intf_pro.wav'))
    print(dsb_name)
    display(ipd.Audio(str(dsb_name)+ '_source_noise_intf_pro.wav'))
    fig, ax = plt.subplots(3, 1, sharey=True)
    fig.set_size_inches((10,15))
    ax[0].set_title('Unprocessed noisy signal at microphone 1')
    ax[0].plot(istft(stft_matrix[0]))
    ax[1].set_title(mvdr_name)
    ax[1].plot(istft(mvdr_source_intf_noise_pr))
    ax[2].set_title(dsb_name)
    ax[2].plot(istft(dsb_source_intf_noise_pr))
    plt.sca(ax[1])
    plt.ylabel('Amplitude')
    plt.sca(ax[2])
    plt.xlabel('Samples')
    fig.subplots_adjust(hspace=0.15)
    plt.show()

def visualize_mvdr_power_pattern(mvdr, t_index, compute_steering_for_freq, compute_doa, compute_tdoa, compute_tdoa_list, f_ind = False):
    dsb = mvdr
    plt.figure()
    env = dsb.env
    num_points = env.num_points
    fs = int(env.f_max*2)
    K =  int(env.os_fft_bins*2-2)
    r = compute_power_pattern_for_freq_mvdr(dsb, num_points, env, t_index,compute_doa, compute_tdoa, compute_steering_for_freq, num_points = num_points)
    theta = np.asarray([point*2*np.pi/num_points for point in range(num_points)])
    #rot = transforms.Affine2D().rotate_deg(90)
    whole = np.squeeze(np.asarray([compute_power_pattern_for_freq_mvdr(dsb,freq, env, t_index,compute_doa, compute_tdoa, compute_steering_for_freq, num_points = num_points) for freq in np.arange(0,env.f_max+1, env.f_max/(env.os_fft_bins-1))]))
    plt.title('Beam Pattern in dB')
    plt.xticks(range(0,num_points + 1,int(num_points/8)),np.arange(0,361,360/8))
    plt.yticks(env.freqs,np.arange(0,env.f_max+1,1000))
    plt.xlabel('Angle of Arrival')
    plt.ylabel('Frequency')
    plt.pcolormesh(whole)
    plt.colorbar()
    plt.show()
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    line,  = ax.plot(theta, np.abs(r))
    ax.set_rmax(0)
    ax.set_rmin(-20)
    ax.set_rticks([-6,-3, 0])
    ax.set_rlabel_position(1)
    ax.set_theta_offset(0)
    ax.grid(True)
    ax.set_title("Beam Pattern in dB", va='bottom')
    fig.show()
    def update(Hz = np.arange(0,env.f_max+1,250)):

        line.set_ydata(compute_power_pattern_for_freq_mvdr(dsb, Hz, env, t_index,compute_doa, compute_tdoa, compute_steering_for_freq, num_points = num_points))
        fig.canvas.draw()

    def sweep(b):
        if f_ind:
            for t_ind in np.arange(0, 401,5):
                line.set_ydata(compute_power_pattern_for_freq_mvdr(dsb, f_ind, env, t_ind,compute_doa, compute_tdoa, compute_steering_for_freq, num_points = num_points,))
                time.sleep(0.1)
                print("\rTime Step: {} ".format(t_ind), end="")
                sys.stdout.flush()
                fig.canvas.draw()


        else:

            for Hz in np.arange(0,env.f_max+1,100):
                line.set_ydata(compute_power_pattern_for_freq_mvdr(dsb, Hz, env, t_index,compute_doa, compute_tdoa, compute_steering_for_freq, num_points = num_points))
                time.sleep(0.1)
                print("\rFrequency: {} Hz ".format(Hz), end="")
                sys.stdout.flush()
                fig.canvas.draw()


    button2 = widgets.Button(description="Beam-Sweep")
    display(button2)
    button2.on_click(sweep)
    interact(update);


    # 6. Let`s take a look at the power patterns

# Those functions are here just for completion ... you do not have to take a look, but you can
def compute_power_pattern_for_freq_mvdr(beamformer, freq, env, t_index, compute_doa, compute_tdoa, compute_steering_for_freq, num_points = 100):
    doa_list = create_sampling_points(num_points)
    theta_list = [compute_doa(point, env.ULA_center, env.ULA_direction) for point in doa_list ]
    #theta_list = [compute_theta(doa, env.ULA_direction) for doa in doa_list]
    tdoa_list = [compute_tdoa(theta, env.d, env.c) for theta in theta_list]
    freq_index = int(round((env.os_fft_bins-1)*freq/env.f_max))
    pos_steerings = []
    for tdoa in tdoa_list:
        pos_steerings.append(compute_steering_for_freq(env.freqs[freq_index], np.array([tdoa*a for a in range(env.N)])))
    return 10*np.log(np.abs(np.square(np.dot(np.asarray(pos_steerings),np.conjugate(beamformer.weights[t_index, freq_index].T)))))

def create_sampling_points(num_points):
    # sample the unit circle with num points
    unit_circle_points = [np.exp(2*np.pi/num_points*point*1j) for point in range(num_points)]
    # transform those into doas
    doa_list = [np.asarray([np.real(a),np.imag(a)]) for a in unit_circle_points]
    return doa_list

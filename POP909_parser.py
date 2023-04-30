import os
import fnmatch
import pretty_midi as pyd
from cachetools import cached

from utils import Chord, Song

note_to_indx = {'C': 0, 'Cb': 0, 'C#': 0, 'D': 1, 'Db': 1, 'D#': 1, 'E': 2, 'Eb': 2, 'E#': 2, 'F': 3, 'Fb': 3, 'F#': 3, 'G': 4, 'Gb': 4, 'G#': 4, 'A': 5, 'Ab': 5, 'A#': 5, 'B': 6, 'Bb': 6, 'B#': 6}
note_to_pitch = {'C': 0, 'Cb': -1, 'C#': 1, 'D': 2, 'Db': 1, 'D#': 3, 'E': 4, 'Eb': 3, 'E#': 5, 'F': 5, 'Fb': 4, 'F#': 6, 'G': 7, 'Gb': 6, 'G#': 8, 'A': 9, 'Ab': 8, 'A#': 10, 'B': 11, 'Bb': 10, 'B#': 0}
type_to_symbol = {'maj': 'M', 'min': 'm', 'sus': 'M', 'aug': 'M', 'dim': 'm'}
special_note_to_indx = {'Hold': 68, 'Rest': 69}
PITCH_SUBTRACT_CONSTANT = 28

@cached(cache={})
def parse_chord_sequence(root_dir):
    sub_dirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    chord_sequences = []
    for sub_dir in sub_dirs: 
        for f in fnmatch.filter(os.listdir(sub_dir), 'key_audio.txt'):
            key_audio_path = os.path.join(sub_dir, f)
            key_sig, quality = process_key_audio(key_audio_path)
        for f in fnmatch.filter(os.listdir(sub_dir), 'chord_audio.txt'):
            chord_audio_path = os.path.join(sub_dir, f)
            chord_seq = process_chord_audio(chord_audio_path, key_sig, quality)
            chord_sequences.append(chord_seq)
    
    return chord_sequences

@cached(cache={})
def parse_songs(root_dir):
    sub_dirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    songs = []
    for sub_dir in sub_dirs: 
        for f in fnmatch.filter(os.listdir(sub_dir), 'key_audio.txt'):
            key_audio_path = os.path.join(sub_dir, f)
            key_sig, quality = process_key_audio(key_audio_path)
        for f in fnmatch.filter(os.listdir(sub_dir), 'chord_audio.txt'):
            chord_audio_path = os.path.join(sub_dir, f)
            chord_seq = process_chord_audio(chord_audio_path, key_sig, quality)
        for f in fnmatch.filter(os.listdir(sub_dir), '*.mid'):
            midi_path = os.path.join(sub_dir, f)
            data = pyd.PrettyMIDI(midi_path)
            notes = []
            assert data.instruments[0].name == 'MELODY'
            notes.extend(data.instruments[0].notes)

            # Transposing Notes to C Major / A Minor
            assert quality in {'maj', 'min'}
            if quality == 'maj':
                for note in notes:
                    note.pitch = note.pitch - note_to_pitch[key_sig]
            else:
                for note in notes:
                    note.pitch = note.pitch - note_to_pitch[key_sig] - 3

            tempo_unit = _get_tempo_unit(notes)
        song = Song(chord_seq, notes, key_sig, quality, tempo_unit)
        songs.append(song)
    return songs

def _get_tempo_unit(notes):
    minimum = float('inf')
    for note in notes:
        startTime = note.start
        endTime = note.end
        diff = endTime - startTime
        minimum = min(minimum, diff)
    return minimum

# Key signature by 1-7 scale degree
def process_key_audio(key_audio_path):
    file = open(key_audio_path, mode='r')
    file = file.read().split()
    key_sig_arr = file[2].split(':')

    note = key_sig_arr[0]
    quality = key_sig_arr[1]
    return note, quality

def process_chord_audio(chord_audio_path, key_sig, quality):
    file = open(chord_audio_path, 'r')
    file = file.read().split()
    chord_set = []
    
    assert quality in {'maj', 'min'}
    if quality == 'maj':
        diff = note_to_indx[key_sig]
    else:
        diff = note_to_indx[key_sig] - 6
    
    for i in range(0, len(file), 3):
        start_time = file[i]
        end_time = file[i+1]
        chord = file[i+2].split(':')
        if len(chord) < 2: 
            continue
        chord = Chord(chord[0], chord[1], start_time, end_time)
        chord.modified_root = (note_to_indx[chord.chord_root] - diff) % 7 + 1
        for key in type_to_symbol.keys(): 
            if key in chord.chord_quality: 
                chord.modified_quality = type_to_symbol[key]
                break
        # Assumption right now: If quality is not in key_dict, then it is major... is that ok?
        if chord.modified_quality == '':
            chord.modified_quality = 'M'
        chord_set.append(chord)
    return chord_set
    

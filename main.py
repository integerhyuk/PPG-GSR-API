from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import json
import traceback

app = FastAPI()


# Function to classify PPG emotion based on SDNN and RMSSD values
def classify_ppg_emotion(ppg_sdnn, ppg_rmssd):
    if ppg_sdnn > 100 and ppg_rmssd > 50:
        return "Happy"
    elif ppg_sdnn < 50 and ppg_rmssd < 30:
        return "Sad"
    elif ppg_sdnn > 100 and ppg_rmssd < 30:
        return "Angry"
    elif 50 <= ppg_sdnn <= 100 and 30 <= ppg_rmssd <= 50:
        return "Neutral"
    else:
        return "Neutral/Calm"


# Function to classify GSR emotion based on SCR amplitude and frequency
def classify_gsr_emotion(scr_amplitude, scr_frequency):
    if scr_amplitude > 0.5 and scr_frequency > 0.05:
        return "Happy/Angry"
    elif scr_amplitude < 0.3 and scr_frequency < 0.02:
        return "Sad"
    elif 0.3 <= scr_amplitude <= 0.5 and 0.02 <= scr_frequency <= 0.05:
        return "Neutral"
    else:
        return "Neutral/Calm"


# Function to detect SCRs and calculate SCR amplitude and frequency
def analyze_gsr(gsr_data, sampling_rate):
    if len(gsr_data) == 0:
        return 0, 0

    # Detect SCR peaks
    scr_peaks, _ = find_peaks(gsr_data, height=0.1, distance=int(1.0 * sampling_rate))

    # Calculate SCR amplitude
    scr_amplitudes = gsr_data[scr_peaks] - np.mean(gsr_data)
    scr_amplitude = np.mean(scr_amplitudes) if len(scr_amplitudes) > 0 else 0

    # Calculate SCR frequency
    scr_frequency = len(scr_peaks) / (len(gsr_data) / sampling_rate)

    return scr_amplitude, scr_frequency


# Function to combine PPG and GSR emotions
def combine_emotions(ppg_emotion, gsr_emotion):
    if gsr_emotion == "Happy/Angry":
        return gsr_emotion
    else:
        return ppg_emotion


# Function to convert time in the format "minute:second.sample number" to seconds
def convert_time_to_seconds(time_str):
    try:
        minutes, seconds_sample = time_str.split(':')
        seconds, _ = seconds_sample.split('.')
        total_seconds = int(minutes) * 60 + int(seconds)
        return total_seconds
    except (ValueError, AttributeError):
        return None


# FastAPI endpoint to analyze sensor data
@app.post("/analyze_sensors/")
async def analyze_sensors(file: UploadFile = File(...), chunks_json: str = Form(...)):
    try:
        df = pd.read_csv(file.file, sep=';')

        if 'time' not in df.columns:
            print("The 'time' column is missing in the CSV file. Using the index as the time column.")
            df['time'] = df.index

        df['time'] = df['time'].apply(convert_time_to_seconds)
        df = df.dropna(subset=['time'])

        if 'ppg_signal' not in df.columns:
            print("The 'ppg_signal' column is missing in the CSV file. Skipping PPG analysis.")
            df['ppg_signal'] = np.nan

        if 'gsr_signal' not in df.columns:
            print("The 'gsr_signal' column is missing in the CSV file. Skipping GSR analysis.")
            df['gsr_signal'] = np.nan

        chunks_data = json.loads(chunks_json)
        print(f"Received chunks_json: {chunks_json}")

        results = []
        sampling_rate = 100  # Assuming a sampling rate of 100 Hz

        for chunk in chunks_data["chunks"]:
            start_time = chunk["timestamp"][0]
            end_time = chunk["timestamp"][1]
            text = chunk["text"]

            print(f"Processing timestamp: {start_time} to {end_time}")

            ppg_segment = df.loc[(df['time'] >= start_time) & (df['time'] <= end_time), 'ppg_signal'].astype(float)
            gsr_segment = df.loc[(df['time'] >= start_time) & (df['time'] <= end_time), 'gsr_signal'].astype(float)

            ppg_emotion = "Neutral/Calm"
            if not ppg_segment.isnull().all():
                ppg_rr_intervals = np.diff(np.where(np.diff(ppg_segment) > 0)[0])

                if len(ppg_rr_intervals) > 1:
                    ppg_sdnn = np.std(ppg_rr_intervals)
                    ppg_rmssd = np.sqrt(np.mean(np.square(np.diff(ppg_rr_intervals))))
                    ppg_emotion = classify_ppg_emotion(ppg_sdnn, ppg_rmssd)

            gsr_emotion = "Neutral/Calm"
            if not gsr_segment.isnull().all():
                scr_amplitude, scr_frequency = analyze_gsr(gsr_segment.dropna(), sampling_rate)
                gsr_emotion = classify_gsr_emotion(scr_amplitude, scr_frequency)

            combined_emotion = combine_emotions(ppg_emotion, gsr_emotion)

            results.append({
                "timestamp": [chunk["timestamp"][0], chunk["timestamp"][1]],
                "text": text,
                "emotion": combined_emotion
            })

        return {"result": results}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format in chunks_json.")
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error processing sensor data.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
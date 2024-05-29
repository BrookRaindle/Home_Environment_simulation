import pandas as pd
from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

def pre_pro_window(data):
    window_data = data.fillna(method='ffill', limit=5)
    columns_to_include = [col for col in window_data.columns if col in ['outside_temp','outdoor_relative_humidity_sensor']]
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(window_data[columns_to_include]), columns=columns_to_include)
    return features_scaled

def pre_pro_close_win(data):
    data = data.fillna(method='ffill', limit=5)
    columns_to_include = [col for col in data.columns if col in ['indoor_temperature_room','humidity']]
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(data[columns_to_include]), columns=columns_to_include)
    return features_scaled

def pre_pro_heated(data):
    data = data.fillna(method='ffill', limit=5)
    data['indoor_temperature_room'] += 0.5
    columns_to_include = [col for col in data.columns if col in ['indoor_temperature_room','humidity']]
    scaler = StandardScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(data[columns_to_include]), columns=columns_to_include)
    return features_scaled

def split_fit_data(data, label):
    features_scaled = data
    model_data = features_scaled.values.reshape(features_scaled.shape[0], features_scaled.shape[1])
    train_inputs, val_inputs, train_label, val_label = train_test_split(model_data, label, test_size=0.2)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(features_scaled.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(train_inputs, train_label, epochs=1000, batch_size=32, validation_data=(val_inputs, val_label), verbose=1)
    acc = model.evaluate(val_inputs, val_label)
    print("Mean squared Error:", acc)
    return model

def train_system(): 
    window_train_data = pd.read_csv('19039840-BR/windows_open.csv')
    train_data = pd.read_csv('19039840-BR/6com2007-3.csv')
    open_features = pre_pro_window(window_train_data)
    close_features = pre_pro_close_win(train_data)
    heated_features = pre_pro_heated(train_data)

    open_window_model = split_fit_data(open_features, window_train_data['satisfaction'])
    closed_window_model = split_fit_data(close_features, train_data['satisfaction'])
    heated_model = split_fit_data(heated_features, train_data['satisfaction'])

    open_window_model.save("Open_Window_Model")
    closed_window_model.save("Close_Window_Model")
    heated_model.save("Heated_Model")

def test_system():
    test_data = pd.read_csv('19039840-BR\\New_day.csv')

    open_window_satisfaction = []
    closed_window_satisfaction = []

    open_window_model = load_model("Open_Window_Model")
    closed_window_model = load_model("Close_Window_Model")
    heated_model = load_model("Heated_Model")

    # Preprocess entire test data
    test_data_close_features = pre_pro_close_win(test_data)
    test_data_open_features = pre_pro_window(test_data)
    test_data_heated_features = pre_pro_heated(test_data)

    test_data_predicted = test_data.copy()

    # Predict satisfaction for open and closed windows + heated rooms
    if not test_data_open_features.empty:
        open_window_satisfaction = open_window_model.predict(test_data_open_features).flatten()
        test_data_predicted['Open_Window_Satisfaction'] = open_window_satisfaction
        print(open_window_satisfaction)
    if not test_data_close_features.empty:
        closed_window_satisfaction = closed_window_model.predict(test_data_close_features).flatten()
        test_data_predicted['Closed_Window_Satisfaction'] = closed_window_satisfaction
        print(closed_window_satisfaction)
    if not test_data_heated_features.empty:
        heated_satisfaction = heated_model.predict(test_data_heated_features).flatten()
        test_data_predicted['Heated_Satisfaction'] = heated_satisfaction
        print(heated_satisfaction)

    test_data_predicted.to_csv('19039840-BR\\New_day_predicted.csv')

def optimize_machines():
     # Load the predicted data
    predicted_data = pd.read_csv('19039840-BR\\New_day_predicted.csv')

    # Convert time to datetime
    predicted_data['Time'] = pd.to_datetime(predicted_data['Time'])

    # Define time windows
    time_window_15min = pd.Timedelta(minutes=15)
    time_window_30min = pd.Timedelta(minutes=30)
    time_window_4hours = pd.Timedelta(hours=4)

    # Find the best 4-hour windows for heating

    num_time_windows = len(predicted_data) // (4 * 60 // 15)

    average_satisfaction_increase = []
    for i in range(num_time_windows):
        start_index = i * (4 * 60 // 15)
        end_index = start_index + (4 * 60 // 15)
        window_data = predicted_data.iloc[start_index:end_index]
        close_satisfaction_avg = window_data['Closed_Window_Satisfaction'].mean()
        heated_satisfaction_avg = window_data['Heated_Satisfaction'].mean()
        average_satisfaction_increase.append((heated_satisfaction_avg - close_satisfaction_avg, start_index))

    average_satisfaction_increase.sort(reverse=True)

    best_heating_windows = [(predicted_data.loc[average_satisfaction_increase[0][1], 'Time'], predicted_data.loc[average_satisfaction_increase[0][1], 'Time'] + time_window_4hours)]
    for i in range(1, len(average_satisfaction_increase)):
        start_time = predicted_data.loc[average_satisfaction_increase[i][1], 'Time']
        end_time = start_time + time_window_4hours
        is_overlapping = False
        for window_start, window_end in best_heating_windows:
            if start_time < window_end and end_time > window_start:
                is_overlapping = True
                break
        if not is_overlapping:
            best_heating_windows.append((start_time, end_time))
            if len(best_heating_windows) == 2:
                break

    best_opening_closing_windows = []            
    window_opened = False
    for index, row in predicted_data.iterrows():
        start_time = row['Time']
        end_time = start_time + time_window_30min
        if end_time <= predicted_data['Time'].iloc[-1]:
            is_overlapping_with_heating = False
            for heating_window in best_heating_windows:
                if start_time < heating_window[1] and end_time > heating_window[0]:
                    is_overlapping_with_heating = True
                    break
            if not is_overlapping_with_heating:
                window_data = predicted_data[(predicted_data['Time'] >= start_time) & (predicted_data['Time'] < end_time)]
                close_satisfaction_avg = window_data['Closed_Window_Satisfaction'].mean()
                open_satisfaction_avg = window_data['Open_Window_Satisfaction'].mean()
                if open_satisfaction_avg > close_satisfaction_avg:
                    if not window_opened:
                        close_time = start_time + time_window_15min
                        best_opening_closing_windows.append((start_time, close_time))
                        window_opened = True
                else:
                    if window_opened:
                        close_time = start_time + time_window_15min
                        best_opening_closing_windows[-1] = (best_opening_closing_windows[-1][0], close_time)
                        window_opened = False

    # Print the chosen times when the heater is activated and deactivated
    print("Best 4-hour heating timeslots:")
    for start_time, end_time in best_heating_windows:
        print(f"Heater activated: {start_time}")
        print(f"Heater deactivated: {start_time + time_window_4hours}")
        print()

    # Print all times when the window is open
    print("window opening and closing times:")
    for start_time, close_time in best_opening_closing_windows:
        print(f"Window opened: {start_time}")
        print(f"Window closed: {close_time}")

    # Add a new column "optimized_satisfaction" to the dataset
    predicted_data['optimized_satisfaction'] = predicted_data['Closed_Window_Satisfaction']
    for start_time, end_time in best_heating_windows:
        predicted_data.loc[(predicted_data['Time'] >= start_time) & (predicted_data['Time'] < end_time), 'optimized_satisfaction'] = predicted_data['Heated_Satisfaction']
    for start_time, close_time in best_opening_closing_windows:
        predicted_data.loc[(predicted_data['Time'] >= start_time) & (predicted_data['Time'] <= close_time), 'optimized_satisfaction'] = predicted_data['Open_Window_Satisfaction']

    for index, row in predicted_data.iterrows():
        time = row['Time']
        optimized_satisfaction = row['optimized_satisfaction']
        in_heating_period = any(start_time <= time < end_time for start_time, end_time in best_heating_windows)
        in_open_window = any(start_time <= time < close_time for start_time, close_time in best_opening_closing_windows)
        print(f"Time: {time}, Satisfaction: {optimized_satisfaction}, Heating: {in_heating_period}, Open Window: {in_open_window}")
        print(f"-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")


    # Save the modified dataset
    predicted_data.to_csv('19039840-BR\\New_day_predicted.csv', index=False)

    # Plot the satisfaction levels
def plot_satisfaction(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data['Closed_Window_Satisfaction'], label='Closed Window Satisfaction')
    plt.plot(data['Time'], data['Open_Window_Satisfaction'], label='Open Window Satisfaction')
    plt.plot(data['Time'], data['Heated_Satisfaction'], label='Heated Satisfaction')
    plt.plot(data['Time'], data['optimized_satisfaction'], label='Optimized Satisfaction')
    plt.xlabel('Time')
    plt.ylabel('Satisfaction')
    plt.legend()
    plt.show()

#train_system()
test_system()
optimize_machines()
plot_satisfaction(pd.read_csv('19039840-BR\\New_day_predicted.csv'))

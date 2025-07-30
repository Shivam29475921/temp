from flask import Flask, render_template
import re
import pandas as pd
from io import StringIO
import json
from collections import Counter
import numpy as np


def analyze_chat_streaks(df):
    longest_streaks = {'Dhruv': 0, 'Kavu': 0}
    longest_gaps = {'Dhruv': pd.Timedelta(0), 'Kavu': pd.Timedelta(0)}
    last_time = {'Dhruv': None, 'Kavu': None}
    current_streak = {'Dhruv': pd.Timedelta(0), 'Kavu': pd.Timedelta(0)}

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        sender = row['sender']
        if sender not in ['Dhruv', 'Kavu']:
            continue
        time_diff = row['timestamp'] - prev['timestamp']
        if row['sender'] == prev['sender']:
            if time_diff <= pd.Timedelta(minutes=10):
                current_streak[sender] += time_diff
            else:
                longest_streaks[sender] = max(longest_streaks[sender], current_streak[sender].total_seconds())
                current_streak[sender] = pd.Timedelta(0)
        else:
            longest_streaks[prev['sender']] = max(longest_streaks[prev['sender']],
                                                  current_streak[prev['sender']].total_seconds())
            current_streak[prev['sender']] = pd.Timedelta(0)

        if last_time[sender]:
            gap = row['timestamp'] - last_time[sender]
            if gap > longest_gaps[sender]:
                longest_gaps[sender] = gap
        last_time[sender] = row['timestamp']

    for sender in ['Dhruv', 'Kavu']:
        longest_streaks[sender] = max(longest_streaks[sender], int(current_streak[sender].total_seconds()))

    return {s: round(longest_streaks[s] / 60, 1) for s in longest_streaks}, {
        s: round(longest_gaps[s].total_seconds() / 3600, 1) for s in longest_gaps}


app = Flask(__name__)


def whatsapp_chat_to_dataframe(chat_data):
    """
    Parses WhatsApp chat data from a string and converts it into a clean pandas DataFrame.
    """
    pattern = re.compile(r'^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}\s?[AP]M) - ')
    chat_lines = StringIO(chat_data).readlines()

    parsed_messages = []
    current_message_data = None

    for line in chat_lines:
        if pattern.match(line):
            if current_message_data:
                parsed_messages.append(current_message_data)

            try:
                timestamp_str = line.split(' - ')[0]
                rest_of_line = ' - '.join(line.split(' - ')[1:])
                sender, message = rest_of_line.split(': ', 1)

                cleaned_sender = sender.strip()
                if 'shivamdas' in cleaned_sender.lower():
                    cleaned_sender = 'Dhruv'
                elif 'labanya' in cleaned_sender.lower():
                    cleaned_sender = 'Kavu'

                current_message_data = {
                    'timestamp': timestamp_str,
                    'sender': cleaned_sender,
                    'message': message.strip()
                }
            except ValueError:
                current_message_data = {
                    'timestamp': timestamp_str,
                    'sender': 'System',
                    'message': rest_of_line.strip()
                }
        else:
            if current_message_data:
                current_message_data['message'] += '\n' + line.strip()

    if current_message_data:
        parsed_messages.append(current_message_data)

    df = pd.DataFrame(parsed_messages)

    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%y, %I:%M %p', errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    df = df[df['sender'] != 'System'].copy()
    df.reset_index(drop=True, inplace=True)

    return df


# --- All Analysis Functions ---

def analyze_chat_details(df):
    """Performs detailed behavioral analysis."""
    hate_you_pattern = re.compile(r'i hate you|hate you', re.IGNORECASE)
    love_you_pattern = re.compile(r'i love you', re.IGNORECASE)
    bhalobashi_pattern = re.compile(r'bhalobashi', re.IGNORECASE)
    ahaa_pattern = re.compile(r'\ba+ha+n+\b', re.IGNORECASE)

    senders = df['sender'].unique()
    analysis = {
        sender: {'message_count': 0, 'hate_you_count': 0, 'love_you_count': 0, 'bhalobashi_count': 0, 'ahaan_count': 0,
                 'nickname_count': 0, 'question_count': 0} for sender in senders}

    for index, row in df.iterrows():
        sender = row['sender']
        message_lower = row['message'].lower()

        if sender in analysis:
            analysis[sender]['message_count'] += 1
            analysis[sender]['bhalobashi_count'] += len(bhalobashi_pattern.findall(message_lower))
            analysis[sender]['ahaan_count'] += len(ahaa_pattern.findall(message_lower))
            analysis[sender]['hate_you_count'] += len(hate_you_pattern.findall(message_lower))
            analysis[sender]['love_you_count'] += len(love_you_pattern.findall(message_lower))
            analysis[sender]['question_count'] += message_lower.count('?')

            if sender == 'Dhruv' and 'kavu' in message_lower:
                analysis[sender]['nickname_count'] += 1
            if sender == 'Kavu' and 'dhruv' in message_lower:
                analysis[sender]['nickname_count'] += 1

    return analysis


def analyze_common_words(df):
    """Finds the top 10 most common words used by both people."""
    senders = df['sender'].unique()
    if len(senders) < 2: return []

    word_counters = {sender: Counter() for sender in senders}
    stop_words = {'the', 'a', 'an', 'in', 'to', 'is', 'i', 'it', 'you', 'and', 'or', 'of', 'for', 'on', 'with', 'that',
                  'this', 'me', 'my', 'your', 'be', 'so', 'are', 'was', 'but', 'have', 'do', 'not', 'at', 'we', 'he',
                  'she', 'all', 'just', 'like', 'get', 'if', 'what', 'can', 'will', 'go', 'no', 'ki', 'r', 'er', 'o',
                  'na', 'ar', 'toh', 'ta', 'je', 'ami', 'tui'}

    for index, row in df.iterrows():
        sender = row['sender']
        message_lower = row['message'].lower()
        if sender in word_counters:
            words = re.findall(r'\b\w+\b', message_lower)
            word_counters[sender].update(w for w in words if w not in stop_words and not w.isdigit())

    sender_list = list(senders)
    set1 = set(word_counters[sender_list[0]].keys())
    set2 = set(word_counters[sender_list[1]].keys())
    common_word_set = set1.intersection(set2)

    common_word_freq = [(word, word_counters[sender_list[0]][word] + word_counters[sender_list[1]][word]) for word in
                        common_word_set]
    common_word_freq.sort(key=lambda x: x[1], reverse=True)
    res = []
    for key, val in common_word_freq[3:20]:
        if len(key) >= 2: res.append((key, val))
    return res[:10]


def analyze_reactions(df):
    """Analyzes who makes the other person laugh or mad."""
    laugh_emoji, mad_emoji = '😂', '👍'
    makes_laugh, makes_mad = Counter(), Counter()
    for i in range(1, len(df)):
        if df['sender'].iloc[i] != df['sender'].iloc[i - 1]:
            if laugh_emoji in df['message'].iloc[i]: makes_laugh[df['sender'].iloc[i - 1]] += 1
            if mad_emoji in df['message'].iloc[i]: makes_mad[df['sender'].iloc[i - 1]] += 1
    return makes_laugh, makes_mad


def analyze_dear_letters_by_month(df):
    """Counts long, loving texts by month."""
    LETTER_LENGTH_THRESHOLD = 150
    letters_df = df[df['message'].str.len() > LETTER_LENGTH_THRESHOLD].copy()
    letters_df['month'] = letters_df['timestamp'].dt.month
    dhruv_letters = letters_df[
        (letters_df['sender'] == 'Kavu') & (letters_df['message'].str.contains('dear dhruv', case=False))]
    kavu_letters = letters_df[
        (letters_df['sender'] == 'Dhruv') & (letters_df['message'].str.contains('dear kavu', case=False))]
    dhruv_counts, kavu_counts = dhruv_letters.groupby('month').size(), kavu_letters.groupby('month').size()
    return [int(dhruv_counts.get(i, 0)) for i in range(1, 13)], [int(kavu_counts.get(i, 0)) for i in range(1, 13)]


def analyze_conversation_starters_and_doubles(df):
    """Analyzes who starts conversations and who double texts."""
    starters, double_texters = Counter(), Counter()
    if not df.empty:
        starters[df['sender'].iloc[0]] += 1
    conversation_start_threshold = pd.Timedelta(hours=4)
    for i in range(1, len(df)):
        if (df['timestamp'].iloc[i] - df['timestamp'].iloc[i - 1]) > conversation_start_threshold:
            starters[df['sender'].iloc[i]] += 1
        if df['sender'].iloc[i] == df['sender'].iloc[i - 1]:
            double_texters[df['sender'].iloc[i]] += 1
    return starters, double_texters


def analyze_chat_heatmap(df):
    """Generates data for the daily chat intensity heatmap from Aug 2023."""
    daily_counts = df.resample('D', on='timestamp').size()
    start_date = pd.to_datetime('2023-08-01')
    end_date = df['timestamp'].max()
    if start_date < df['timestamp'].min():
        start_date = df['timestamp'].min()
    date_range = pd.date_range(start=start_date.normalize(), end=end_date.normalize())
    heatmap_data = [{'date': str(date.date()), 'count': int(daily_counts.get(date, 0))} for date in date_range]
    return heatmap_data


def analyze_rolling_average(df):
    """Calculates the 7-day rolling average of message counts."""
    daily_counts = df.resample('D', on='timestamp').size()
    rolling_avg = daily_counts.rolling(window=7, min_periods=1).mean().round(1)
    return list(rolling_avg.index.strftime('%Y-%m-%d')), list(rolling_avg.values)


# --- Flask Route ---
@app.route('/')
def dashboard():
    """Renders the main analysis dashboard."""
    try:
        with open('chat.txt', 'r', encoding='utf-8') as f:
            chat_data = f.read()
    except FileNotFoundError:
        return "Error: 'WhatsApp Chat with Labanya 3.txt' not found."

    df = whatsapp_chat_to_dataframe(chat_data)

    details_analysis = analyze_chat_details(df)
    common_words = analyze_common_words(df)

    most_common_dhruv = Counter()
    most_common_kavu = Counter()
    for index, row in df.iterrows():
        message_lower = row['message'].lower()
        words = re.findall(r'\b\w+\b', message_lower)
        filtered_words = [w for w in words if
                          w not in {'the', 'a', 'an', 'in', 'to', 'is', 'i', 'it', 'you', 'and', 'or', 'of', 'for',
                                    'on', 'with', 'that', 'this', 'me', 'my', 'your', 'be', 'so', 'are', 'was', 'but',
                                    'have', 'do', 'not', 'at', 'we', 'he', 'she', 'all', 'just', 'like', 'get', 'if',
                                    'what', 'can', 'will', 'go', 'no', 'ki', 'r', 'er', 'o', 'na', 'ar', 'toh', 'ta',
                                    'je', 'ami', 'tui'} and not w.isdigit()]
        if row['sender'] == 'Dhruv':
            most_common_dhruv.update(filtered_words)
        elif row['sender'] == 'Kavu':
            most_common_kavu.update(filtered_words)

    top_dhruv_words = [el for el in most_common_dhruv.most_common(20)[3:14] if len(el[0]) >= 3]
    top_kavu_words = most_common_kavu.most_common(10)[3:10]

    conversation_starters, double_texts = analyze_conversation_starters_and_doubles(df)
    makes_laugh, makes_mad = analyze_reactions(df)
    dhruv_letter_counts, kavu_letter_counts = analyze_dear_letters_by_month(df)
    heatmap_data = analyze_chat_heatmap(df)
    rolling_avg_dates, rolling_avg_values = analyze_rolling_average(df)
    streaks, gaps = analyze_chat_streaks(df)

    love_score_dates = list(df.resample('D', on='timestamp').size().index.strftime('%Y-%m-%d'))
    love_score_values = [100] * len(love_score_dates)

    labels = [sender for sender in details_analysis if isinstance(details_analysis.get(sender), dict)]

    message_counts = [details_analysis.get(label, {}).get('message_count', 0) for label in labels]
    nickname_counts = [details_analysis.get(label, {}).get('nickname_count', 0) for label in labels]
    laugh_counts = [makes_laugh.get(label, 0) for label in labels]
    mad_counts = [makes_mad.get(label, 0) for label in labels]
    question_counts = [details_analysis.get(label, {}).get('question_count', 0) for label in labels]
    starter_counts = [conversation_starters.get(label, 0) for label in labels]
    double_text_counts = [double_texts.get(label, 0) for label in labels]

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return render_template('dashboard.html',
                           analysis=details_analysis,
                           labels=json.dumps(labels),
                           message_counts=json.dumps(message_counts),
                           nickname_counts=json.dumps(nickname_counts),
                           laugh_counts=json.dumps(laugh_counts),
                           mad_counts=json.dumps(mad_counts),
                           question_counts=json.dumps(question_counts),
                           starter_counts=json.dumps(starter_counts),
                           double_text_counts=json.dumps(double_text_counts),
                           dhruv_letter_counts=json.dumps(dhruv_letter_counts),
                           kavu_letter_counts=json.dumps(kavu_letter_counts),
                           month_labels=json.dumps(month_labels),
                           heatmap_data=json.dumps(heatmap_data),
                           common_words=json.dumps(common_words),
                           rolling_avg_dates=json.dumps(rolling_avg_dates),
                           rolling_avg_values=json.dumps(rolling_avg_values),
                           love_score_dates=json.dumps(love_score_dates),
                           love_score_values=json.dumps(love_score_values),
                           longest_streaks=json.dumps(streaks),
                           longest_gaps=json.dumps(gaps),
                           top_dhruv_words=json.dumps(top_dhruv_words),
                           top_kavu_words=json.dumps(top_kavu_words))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tensorboard import summary
import wikipedia
import PySimpleGUI as sg
from io import StringIO
import sys
import traceback



def getTable():
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    table_classes = [table.get("class", []) for table in tables]
    th_names = [
        table.find_all("th")[0].text if table.find_all("th") else "" for table in tables
    ]
    merged_list = [
        (table_classes[i], th_names[i]) for i in range(0, len(table_classes))
    ]

    layout = [
        [sg.Text("Select a Table:")],
        [sg.InputCombo(merged_list, key="combo")],
        [sg.Button("OK"), sg.Button("Cancel")],
    ]

    window = sg.Window("Table Select", layout, location=(100, 100))
    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "Cancel"):
            break
        elif event == "OK":
            selected_option = values["combo"]
            break

    table_scrape = soup.find("table", {"class": selected_option})

    df = pd.read_html(str(table_scrape))
    df = pd.DataFrame(df[0])
    df.to_csv(csv_path)
    sg.popup('Data Saved!', title='Table Saved')
    
def wikiSummary():
    content = wikipedia.WikipediaPage(title = page_title).summary
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write(content)
    sg.popup(content, title='Summary')
    
def on_dropdown_change(value):
    if value == 'Table':
        getTable()
    if value == 'Summary':
        wikiSummary()
    
layout = [
    [sg.Text("Wiki Page Title:"), sg.InputText(key="input1")],
    [sg.Text("CSV Path:"), sg.InputText(key="input2")],
    [sg.Text("Select Element"), sg.Combo(['Table','Summary'], key='-DROPDOWN-', enable_events=True)],
    [sg.Button("Scrape"), sg.Button("Exit")],
]

window = sg.Window("Wiki Scraper", layout)

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "Scrape":
        page_title = values["input1"]
        csv_path = values["input2"]
        selected_option = values['-DROPDOWN-']
        wikiurl = wikipedia.page(page_title, auto_suggest=False).url
        response = requests.get(wikiurl)
        print(response.status_code)
        on_dropdown_change(selected_option)        
        break
    
window.close()
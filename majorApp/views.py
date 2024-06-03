from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.urls import reverse
import yfinance as yf
import pandas as pd
import csv


def searchPage(request):
    csv_file = 'majorApp/static/CSV/companies.csv'  # Path to your CSV file
    options = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            options.append({'value': row['SYMBOL'], 'text': row['NAME OF COMPANY']})

    return render(request, 'majorApp/searchPage.html', {'options': options})


def searching(request):
    if request.method == 'POST':
        c_name = request.POST.get('c_name')
        cname = request.session['cname'] = c_name
        print(c_name)
        if len(c_name) == 0:
            alert_message = True
            return render(request, 'majorApp/searchPage.html', {'alert_message': alert_message})
        else:
            c_name += ".NS"
            name = yf.Ticker(c_name)
            context = {'summary': get_summary(c_name).replace(';', ','),
                       'c_name': c_name,
                       'first_officer_info': get_officers_info(c_name),
                       'industry': name.info.get('industry'),
                       'sector': name.info.get('sector'),
                       'website': name.info.get('website')}
            return render(request, 'majorApp/overview.html', context)
    return render(request, 'majorApp/searchPage.html')


def navigation(request):
    if request.method == 'POST':
        cname = request.session.get('cname')
        action = request.POST.get('task')
        if action == 'overview':
            cname += ".NS"
            name = yf.Ticker(cname)
            context = {'summary': get_summary(cname).replace(';', ','),
                       'c_name': cname,
                       'first_officer_info': get_officers_info(cname),
                       'industry': name.info.get('industry'),
                       'sector': name.info.get('sector'),
                       'website': name.info.get('website')}
            return render(request, 'majorApp/overview.html', context)
        elif action == 'financials':
            context = {'company_name': cname}
            return render(request, 'majorApp/financials.html', context)
        elif action == 'fundamentals':
            context = {'company_name': cname}
            return render(request, 'majorApp/fundamentals.html', context)
        elif action == 'projections':
            return render(request, 'majorApp/projections.html')
        elif action == 'home':
            return render(request, 'majorApp/searchPage.html')
        else:
            alert_message = True
            return render(request, 'majorApp/searchPage.html', {'alert_message': alert_message})
    return render(request, 'majorApp/navigation.html')


def get_summary(cname):
    name = yf.Ticker(cname)
    return name.info.get('longBusinessSummary')


def get_officers_info(cname):
    name = yf.Ticker(cname)
    officers = name.info.get('companyOfficers', [])
    officers_info = [{'name': officer.get('name'), 'title': officer.get('title')} for officer in officers]
    first_officer_info = officers_info[0]
    return first_officer_info


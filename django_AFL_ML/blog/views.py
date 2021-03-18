from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

posts = [
    {
        'author': 'Mitch',
        'title': 'First Post',
        'content': 'First Post content',
        'date_posted': '18th March, 2021',
    },
    {
        'author': 'Bec',
        'title': '2nd Post',
        'content': '2nd Post content',
        'date_posted': '19th March, 2021',
    }
]

context = {
    'title': 'V.0'
}

def home(request):
    return render(request, 'blog/home.html', context)

def about(request):
    return render(request, 'blog/about.html', context)

def history(request):
    return render(request, 'blog/history.html', context)

def discussion(request):
    context = {
        'posts': posts,
        'title': 'V.0'
    }
    return render(request, 'blog/discussion.html',context)

from django import template

register = template.Library()


@register.filter
def human_readable_large_number(value):
    try:
        value = float(value)/1e7
        return f'{value:.2f} Cr'
    except (ValueError, TypeError):
        return value


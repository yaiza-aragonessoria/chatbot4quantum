# Generated by Django 4.2.7 on 2024-03-11 11:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('message', '0011_message_needs_button'),
    ]

    operations = [
        migrations.AlterField(
            model_name='message',
            name='needs_button',
            field=models.CharField(choices=[('False', 'false'), ('ok', 'ok'), ('compute', 'compute')], default='false', max_length=20),
        ),
    ]

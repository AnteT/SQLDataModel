import sqlite3, datetime

def register_adapters_and_converters():
    def adapt_date(val):
        """unchanged from sqlite3 default adapters"""
        return val.isoformat()

    def adapt_datetime(val):
        """unchanged from sqlite3 default adapters"""
        return val.isoformat(" ")

    def convert_date(val):
        """modified to avoid ValueError on parsing date from datetime and restrict input to first 10 items of val"""
        return datetime.date(*map(int, val[:10].split(b"-")))

    def convert_timestamp(val):
        """modified to avoid ValueError from parsing datetime from date and provide default timepart to val"""
        if len(val) <= 10:
            datepart, timepart = val, b'00:00:00'
        else:
            datepart, timepart = val.split(b" ")
        year, month, day = map(int, datepart.split(b"-"))
        timepart_full = timepart.split(b".")
        hours, minutes, seconds = map(int, timepart_full[0].split(b":"))
        if len(timepart_full) == 2:
            microseconds = int('{:0<6.6}'.format(timepart_full[1].decode()))
        else:
            microseconds = 0
        val = datetime.datetime(year, month, day, hours, minutes, seconds, microseconds)
        return val

    sqlite3.register_adapter(datetime.date, adapt_date)
    sqlite3.register_adapter(datetime.datetime, adapt_datetime)
    sqlite3.register_converter("date", convert_date)
    sqlite3.register_converter("timestamp", convert_timestamp)

  
# TALES UNIT TESTS

`test_app.py` 
This is largely a copy of the test used in simple_weather.

`test_walkthrough.py`
Tests extracting the walkthrough and stepping through the gold trajectory actions for all frameworks in TALES.

For walkthroughs that are larger than 100 steps, we play the actions up to the 100th walkthrough action and report the fraction of the max score achieved. 
from typing import Union

from fastapi import FastAPI
import mlflow.spark
import pandas as pd
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType

model_uri='/home/t814955/best-model'

#define the hour zone
a = range(24)
HrZone = {x: 'Morning' for x in a[:6]}
HrZone.update({x: 'MidDay' for x in a[6:18]})
HrZone.update({x: 'Evening' for x in a[18:]})

# Define the schema for the input data
Input_schema = StructType([
    StructField("trip", StringType(), True),
    StructField("hour", IntegerType(), True),
    StructField("HrZone", StringType(), True),
    StructField("counts", LongType(), True)
    ])

#spark session 
spark = SparkSession.builder \
    .master('spark://qs2:7077') \
    .appName("API_App") \
    .config("spark.executor.memory", "10g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

#load model 
model = mlflow.spark.load_model(model_uri=model_uri)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/stop")
def stop_spark():
    mlflow.end_run()
    spark.stop()
    return {"Stop": "Spark"}

@app.get("/traffic/{hour}")
async def read_traffic_volume(hour: int, src_dst: Union[str, None] = None):
    try:
        counts = '-1'
    
        trip = '{}_{}'.format(*src_dst.split(','))
        ltrip = ["237_236", "264_264", "236_237", "237_237", "236_236", "237_161", "161_237",
                 "161_236", "239_142", "142_239", "239_238", "141_236", "236_161"]
        if not trip in ltrip:
            return {"hour": hour, "counts": counts}
    
        df_prepped =  spark.createDataFrame(
            pd.DataFrame(data = {'trip' : [trip], 'hour' : [hour], 'HrZone' : [HrZone[hour]], 'counts' : [0]}), 
            schema=Input_schema)
    
        target = 'counts'
        holdout = model.transform(df_prepped).select([target, 'prediction']).toPandas()
        counts=round(holdout.loc[0, 'prediction'])
    
        return {"hour": hour, "counts": str(counts)}
    except Exception as e:
        return {"error" : str(e)}
    

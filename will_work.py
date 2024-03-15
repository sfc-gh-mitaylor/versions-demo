# The Snowpark package is required for Python Worksheets. 
# You can add more packages by selecting them using the Packages control and then importing them.

import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col
from snowflake.snowpark.functions import lit
from snowflake.ml.modeling.ensemble import RandomForestClassifier as RFC
from snowflake.ml.registry import Registry

def main(session: snowpark.Session):

    #Creating a Dummy Dataset
    session.sql("CREATE OR REPLACE DATABASE DEMO_DB").collect()
    session.sql("CREATE OR REPLACE SCHEMA PUBLIC").collect()    
    from sklearn.datasets import make_classification
    import pandas as pd
    columns = [str(i) for i in range(0,5)]
    X,y = make_classification(n_samples=100, n_features=5, n_classes=2)
    df = pd.DataFrame(X, columns=columns)
    df['Y'] = y
    session.write_pandas(df, table_name='DUMMY_DATA', auto_create_table=True, overwrite=True)
    df = session.table('DUMMY_DATA')

    # Prep and Train
    rf = RFC(input_cols=df.columns[:5], label_cols="Y", output_cols="PREDICT")
    rf.fit(df)

    # 
    session.sql("CREATE OR REPLACE DATABASE REGISTRY_DB").collect()
    session.sql("CREATE OR REPLACE SCHEMA PUBLIC").collect()
    reg = Registry(session=session, database_name="REGISTRY_DB", schema_name="PUBLIC")
    model_ver = reg.log_model(
        rf,
        model_name="RFC_TEST",
        version_name="v1",
        conda_dependencies=["scikit-learn==1.2.2", # unless specified the dependencies will choose the latest version in the conda channel, not the version in local env.
                            'snowflake-ml-python==1.2.0'], 
        comment="Test",
        metrics={},
        sample_input_data=df.columns[:5])

    session.sql("USE DATABASE DEMO_DB").collect()
    session.sql("USE SCHEMA PUBLIC").collect()
    df = session.table('DUMMY_DATA')
    res_df = rf.predict(df)
    
    return res_df.show()
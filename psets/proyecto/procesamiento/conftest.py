"""pytest no recoge tests/test_build_obt.py.

Esa prueba es del job de Spark: se corre con spark-submit dentro del contenedor
spark-master (ver README.md), no con pytest. En GitHub Actions no hay pyspark, asi
que al importarla `pytest psets/proyecto` fallaria antes de correr las demas pruebas.
"""

collect_ignore = ["tests/test_build_obt.py"]

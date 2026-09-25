pip install -r requirements.txt

# for f in \
#     FusionEngine/fused_decisions_chunk3_test.csv \
#     PositionSizing/position_sizing_chunk3_test.csv \
#     QuantitativeAnalyst/quantitative_analysis_chunk3_test.csv
# do
#     split -b 45M -d --suffix-length=3 "$f" "${f%.csv}_"
# done

cd outputs/resultas/PositionSizing
cat position_sizing_chunk3_test_* > position_sizing_chunk3_test.csv

cd ../FusionEngine
cat position_sizing_chunk3_test_* > position_sizing_chunk3_test.csv

cd ../QuantitativeAnalyst
cat quantitative_analysis_chunk3_test_* > quantitative_analysis_chunk3_test.csv

cd ../../..
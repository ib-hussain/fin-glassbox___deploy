pip install --extra-index-url https://download.pytorch.org/whl/cpu torch>=2.7.1+cpu
pip install numpy pandas transformers streamlit

# for f in \
#     FusionEngine/fused_decisions_chunk3_test.csv \
#     PositionSizing/position_sizing_chunk3_test.csv \
#     QuantitativeAnalyst/quantitative_analysis_chunk3_test.csv
# do
#     split -b 45M -d --suffix-length=3 "$f" "${f%.csv}_"
# done

cd outputs/results/PositionSizing/
cat position_sizing_chunk3_test_* > position_sizing_chunk3_test.csv

cd ../FusionEngine/
cat position_sizing_chunk3_test_* > position_sizing_chunk3_test.csv

cd ../QuantitativeAnalyst/
cat quantitative_analysis_chunk3_test_* > quantitative_analysis_chunk3_test.csv

cd ../../..
# Data Description

## File Description

- `categorical-numeric-mappings.yml`: the mapping from categorical literal value to encoding value; 
- `census.csv`: the _manually_ proccessed monolithic csv data sheet from source data files;
- `import-census-data.sql`: the SQL commands importing the csv file to PostgreSQL database, for accelerating the development process.

## Data Table Description

Note: The order of attributes description is rearranged according to their groups (categories)

| Attribute | Category | Data type | Description |
|---|---|---|---|
| YEAR | Temporal | Categorical | Census survey year (1994 or 1995). |
| AAGE | Demographic | Integer | Age of the individual. |
| AMARITL | Demographic | Categorical | Marital status. |
| AREORGN | Demographic | Categorical | Hispanic origin. |
| ASEX | Demographic | Categorical | Sex. |
| ARACE | Demographic | Categorical | Racial group. |
| PENATVTY | Demographic | Categorical | Individual's country of birth. |
| PRCITSHP | Demographic | Categorical | Citizenship status. |
| ACLSWKR | Employment | Categorical | Class of worker. |
| ADTIND | Employment | Categorical | Industry code of the individual's employer. |
| ADTOCC | Employment | Categorical | Occupation code of the individual. |
| AMJIND | Employment | Categorical | Major industry classification. |
| AMJOCC | Employment | Categorical | Major occupation classification. |
| AUNMEM | Employment | Categorical | Labor union membership status. |
| AUNTYPE | Employment | Categorical | Reason for unemployment. |
| AWKSTAT | Employment | Categorical | Full-time or part-time employment status. |
| NOEMP | Employment | Integer | Number of persons working for the employer. |
| SEOTR | Employment | Categorical | Own-business or self-employment status. |
| WKSWORK | Employment | Integer | Number of weeks worked during the year. |
| AHGA | Education | Categorical | Highest educational attainment. |
| AHSCOL | Education | Categorical | Educational institution enrollment status during the previous week. |
| HHDFMX | Household | Categorical | Detailed household and family status. |
| HHDREL | Household | Categorical | Individual's relationship to the household. |
| PARENT | Household | Categorical | Family members under 18 and their parental presence. |
| PEFNTVTY | Family background | Categorical | Father's country of birth. |
| PEMNTVTY | Family background | Categorical | Mother's country of birth. |
| AHRSPAY | Financial | Integer | Wage per hour. |
| CAPGAIN | Financial | Integer | Capital gains. |
| CAPLOSS | Financial | Integer | Capital losses. |
| DIVVAL | Financial | Integer | Dividends from stocks. |
| FILESTAT | Financial | Categorical | Federal income tax filing status. |
| GRINREG | Migration | Categorical | Region of previous residence. |
| GRINST | Migration | Categorical | State of previous residence. |
| MIGMTR1 | Migration | Categorical | Migration status based on changes in metropolitan statistical area (MSA). |
| MIGMTR3 | Migration | Categorical | Migration status based on changes in region. |
| MIGMTR4 | Migration | Categorical | Migration status based on moves within a region. |
| MIGSAME | Migration | Categorical | Whether the individual lived in the same house one year ago. |
| MIGSUN | Migration | Categorical | Whether the previous residence was in the Sunbelt region. |
| VETQVA | Veteran status | Categorical | Whether the individual completed an income questionnaire for the Veterans Administration. |
| VETYN | Veteran status | Categorical | Veterans' benefits status. |
| MARSUPWT | Sampling | Integer | Instance weight representing the population size associated with each observation. Excluded from classifier predictors. |
| income | Target | Categorical | Binary income category: below $50,000 or $50,000 and above. |
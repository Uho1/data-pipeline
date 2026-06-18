from __future__ import annotations

import re

import pandas as pd

from market_data.utils import now_utc_iso

_REPORT_META = {
    "11013": {
        "term": "Q1",
        "month": 3,
        "day": 31,
        "form_type": "Q1",
        "order": 1,
        "quarter_start": (1, 1),
    },
    "11012": {
        "term": "Q2",
        "month": 6,
        "day": 30,
        "form_type": "H1",
        "order": 2,
        "quarter_start": (4, 1),
    },
    "11014": {
        "term": "Q3",
        "month": 9,
        "day": 30,
        "form_type": "Q3",
        "order": 3,
        "quarter_start": (7, 1),
    },
    "11011": {
        "term": "Q4",
        "month": 12,
        "day": 31,
        "form_type": "FY",
        "order": 4,
        "quarter_start": (10, 1),
    },
}

_ACCOUNT_MAP = {
    "ifrs-full_Revenue": "Revenue",
    "ifrs-full_InsuranceRevenue": "Revenue",
    "ifrs-full_InsuranceContractsIssuedThatAreAssets": "Insurance Contract Assets",
    "ifrs-full_ReinsuranceContractsHeldThatAreAssets": "Reinsurance Contract Assets",
    "ifrs-full_RevenueFromInterest": "Interest Revenue",
    "ifrs-full_InterestRevenueCalculatedUsingEffectiveInterestMethod": "Effective Interest Revenue",
    "ifrs-full_InterestRevenueForFinancialAssetsMeasuredAtFairValueThroughOtherComprehensiveIncome": "Effective Interest Revenue",
    "ifrs-full_InterestIncomeOnFinancialAssetsDesignatedAtFairValueThroughProfitOrLoss": "FVTPL Interest Revenue",
    "ifrs-full_InterestRevenueExpense": "Net Interest Revenue",
    "ifrs-full_InterestExpense": "Interest Expense",
    "ifrs-full_InvestmentIncome": "Investment Income",
    "dart_InvestmentIncomeExpenses": "Investment Gain/Loss",
    "ifrs-full_FeeAndCommissionIncome": "Fee Income",
    "ifrs-full_FeeAndCommissionIncomeExpense": "Net Fee Income",
    "ifrs-full_FeeAndCommissionExpense": "Fee Expense",
    "dart_InsuranceRevenueExpense": "Insurance Service Result",
    "ifrs-full_InsuranceServiceResult": "Insurance Service Result",
    "dart_OperatingIncomeInsurance": "Insurance Revenue Component",
    "dart_InsuranceFinanceIncomeFromInsuranceContractsIssuedRecognisedInProfitOrLoss": "Insurance Finance Income",
    "dart_InsuranceFinanceExpensesFromInsuranceContractsIssuedRecognisedInProfitOrLoss": "Insurance Finance Expense",
    "dart_FinanceIncomeFromReinsuranceContractsHeldRecognisedInProfitOrLoss": "Reinsurance Finance Income",
    "dart_FinanceExpensesFromReinsuranceContractsHeldRecognisedInProfitOrLoss": "Reinsurance Finance Expense",
    "dart_OtherOperatingIncome": "Other Operating Income Component",
    "dart_OtherOperatingIncomeInvestment": "Other Operating Income Component",
    "ifrs-full_OtherOperatingIncomeExpense": "Other Operating Income Component",
    "ifrs-full_MiscellaneousOtherOperatingIncome": "Other Operating Income Component",
    "dart_GainLossFromFinancialInstrumentsAtFairValueThroughProfitOrLoss": "Trading Gain",
    "dart_GainLossFromFinancialInstrumentsAtFairValueThroughOtherComprehensiveIncome": "Trading Gain",
    "dart_GainLossFromFinancialInstrumentsAtAmortisedCost": "Trading Gain",
    "dart_GainFromFinancialInstruments": "Trading Gain",
    "dart_GainsOnChangeInFairValueAndOnDisposalOfFinancialInstruments": "Trading Gain",
    "dart_LossesOnChangeInFairValueAndOnDisposalOfFinancialInstruments": "Trading Loss",
    "dart_GainFromFinancialInstrumentsAtFairValueThroughProfitOrLoss": "Trading Gain",
    "dart_GainFromFinancialInstrumentsAtFairValueThroughOtherComprehensiveIncome": "Trading Gain",
    "dart_GainFromFinancialInstrumentsAtAmortisedCost": "Trading Gain",
    "dart_LossFromFinancialInstruments": "Trading Loss",
    "dart_LossFromFinancialInstrumentsAtFairValueThroughProfitOrLoss": "Trading Loss",
    "dart_LossFromFinancialInstrumentsAtFairValueThroughOtherComprehensiveIncome": "Trading Loss",
    "dart_LossFromFinancialInstrumentsAtAmortisedCost": "Trading Loss",
    "ifrs-full_NetForeignExchangeGain": "Trading Gain",
    "ifrs-full_ForeignExchangeGain": "Trading Gain",
    "ifrs-full_ForeignExchangeLoss": "Trading Loss",
    "dart_GainFromDerivatives": "Trading Gain",
    "dart_LossesFromDerivatives": "Trading Loss",
    "ifrs-full_CostOfSales": "COGS",
    "ifrs-full_GrossProfit": "Gross Profit",
    "dart_SellingGeneralAdministrativeExpenses": "SG&A",
    "dart_TotalSellingGeneralAdministrativeExpenses": "SG&A",
    "ifrs-full_ResearchAndDevelopmentExpense": "R&D",
    "ifrs-full_OperatingProfitLoss": "Operating Income",
    "ifrs-full_ProfitLossFromOperatingActivities": "Operating Income",
    "dart_OperatingIncomeLoss": "Operating Income",
    "ifrs-full_ProfitLossBeforeTax": "Pretax Income",
    "ifrs-full_ProfitLoss": "Net Income",
    "dart_ProfitLossAttributableToOwnersOfParentEntity": "Net Income Common",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "Net Income Common",
    "ifrs-full_IncomeTaxExpenseContinuingOperations": "Tax",
    "ifrs-full_Assets": "Total Assets",
    "ifrs-full_CurrentAssets": "Current Assets",
    "ifrs-full_Liabilities": "Total Liabilities",
    "ifrs-full_CurrentLiabilities": "Current Liabilities",
    "ifrs-full_Equity": "Shareholders Equity",
    "ifrs-full_CashAndCashEquivalents": "Cash",
    "dart_CashAndDuefromBanks": "Cash",
    "ifrs-full_CurrentFinancialAssetsAtAmortisedCost": "Short-term Investments",
    "ifrs-full_CurrentFinancialAssetsAtFairValueThroughProfitOrLoss": "Short-term Investments",
    "ifrs-full_FinancialAssetsAtFairValueThroughProfitOrLossCurrent": "Short-term Investments",
    "ifrs-full_OtherCurrentFinancialAssets": "Short-term Investments",
    "ifrs-full_CurrentInvestments": "Short-term Investments",
    "ifrs-full_ShorttermDepositsNotClassifiedAsCashEquivalents": "Short-term Investments",
    "ifrs-full_TradeAndOtherCurrentReceivables": "AR",
    "ifrs-full_TradeReceivablesCurrent": "AR",
    "ifrs-full_CurrentTradeReceivables": "AR",
    "dart_ShortTermTradeReceivable": "AR",
    "dart_ShortTermOtherReceivablesNet": "Other Receivables",
    "ifrs-full_OtherCurrentReceivables": "Other Receivables",
    "dart_CurrentNontradeReceivables": "Other Receivables",
    "ifrs-full_CurrentContractAssets": "Contract Assets",
    "dart_ShortTermDueFromCustomersForContractWorkNet": "Construction Receivables",
    "ifrs-full_Inventories": "Inventory",
    "ifrs-full_TradeAndOtherCurrentPayables": "AP",
    "ifrs-full_TradePayablesCurrent": "AP",
    "ifrs-full_ShorttermBorrowings": "Debt Short",
    "ifrs-full_LongtermBorrowings": "Debt Long",
    "ifrs-full_Goodwill": "Goodwill",
    "ifrs-full_IntangibleAssetsOtherThanGoodwill": "Intangibles",
    "ifrs-full_IntangibleAssetsAndGoodwill": "Intangibles",
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "Operating Cash Flow",
    "ifrs-full_CashFlowsFromUsedInInvestingActivities": "Investing Cash Flow",
    "ifrs-full_CashFlowsFromUsedInFinancingActivities": "Financing Cash Flow",
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": "Capital Expenditure",
    "ifrs-full_PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities": "Capital Expenditure",
    "ifrs-full_PurchaseOfInvestmentProperty": "Capital Expenditure",
    "ifrs-full_PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities": "PPE CapEx",
    "ifrs-full_PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities": "Intangible CapEx",
    "ifrs-full_PurchaseOfInvestmentProperty": "Investment Property CapEx",
    "ifrs-full_DividendsPaidClassifiedAsFinancingActivities": "Dividends Paid",
    "ifrs-full_DividendsPaid": "Dividends Paid",
    "ifrs-full_RevenueFromDividends": "Investment Income",
    "ifrs-full_PurchaseOfTreasuryShares": "Repurchases",
    "dart_BasicEarningsPerShare": "EPS",
    "ifrs-full_BasicEarningsLossPerShare": "EPS",
    "ifrs-full_NumberOfSharesOutstanding": "Shares",
    # D&A — Cash Flow Statement adjustment items (간접법 현금흐름표 조정항목)
    "dart_AdjustmentsForDepreciationExpense": "D&A",
    "dart_AdjustmentsForAmortisationExpense": "D&A",
    "dart_AdjustmentsForDepreciationRightofuseAssets": "D&A",
    "ifrs-full_AdjustmentsForDepreciationExpense": "D&A",
    "ifrs-full_AdjustmentsForAmortisationExpense": "D&A",
    "dart_DepreciationExpense": "D&A",
    "dart_AmortisationExpense": "D&A",
    "ifrs-full_DepreciationExpense": "D&A",
    "ifrs-full_AmortisationExpense": "D&A",
    # D&A — SGA/Cost of Sales 하위 항목으로 분류된 경우
    "dart_DepreciationExpenseSellingGeneralAdministrativeExpenses": "D&A",
    "dart_AmortisationExpenseSellingGeneralAdministrativeExpenses": "D&A",
    "dart_DepreciationExpenseCostOfSales": "D&A",
    "dart_AmortisationExpenseCostOfSales": "D&A",
    "ifrs-full_DepreciationExpenseSellingGeneralAdministrativeExpenses": "D&A",
    "ifrs-full_AmortisationExpenseSellingGeneralAdministrativeExpenses": "D&A",
    # D&A — 통합 태그
    "dart_DepreciationAndAmortisationExpense": "D&A",
    "ifrs-full_DepreciationAndAmortisationExpense": "D&A",
    "dart_DepreciationAmortisationExpenseAndImpairmentLoss": "D&A",
}

_ACCOUNT_ID_CORE_MAP = {
    "revenue": "Revenue",
    "revenuefrominterest": "Interest Revenue",
    "interestrevenuecalculatedusingeffectiveinterestmethod": "Effective Interest Revenue",
    "interestrevenueforfinancialassetsmeasuredatfairvaluethroughothercomprehensiveincome": "Effective Interest Revenue",
    "interestincomeonfinancialassetsdesignatedatfairvaluethroughprofitorloss": "FVTPL Interest Revenue",
    "interestrevenueexpense": "Net Interest Revenue",
    "interestexpense": "Interest Expense",
    "investmentincome": "Investment Income",
    "feeandcommissionincome": "Fee Income",
    "feeandcommissionincomeexpense": "Net Fee Income",
    "feeandcommissionexpense": "Fee Expense",
    "insurancerevenueexpense": "Insurance Service Result",
    "insuranceserviceresult": "Insurance Service Result",
    "insurancerevenue": "Insurance Revenue Component",
    "operatingincomeinsurance": "Insurance Revenue Component",
    "insurancecontractsissuedthatareassets": "Insurance Contract Assets",
    "reinsurancecontractsheldthatareassets": "Reinsurance Contract Assets",
    "investmentincomeexpenses": "Investment Gain/Loss",
    "insurancefinanceincomefrominsurancecontractsissuedrecognisedinprofitorloss": "Insurance Finance Income",
    "insurancefinanceexpensesfrominsurancecontractsissuedrecognisedinprofitorloss": "Insurance Finance Expense",
    "financeincomefromreinsurancecontractsheldrecognisedinprofitorloss": "Reinsurance Finance Income",
    "financeexpensesfromreinsurancecontractsheldrecognisedinprofitorloss": "Reinsurance Finance Expense",
    "otheroperatingincome": "Other Operating Income Component",
    "otheroperatingincomeexpense": "Other Operating Income Component",
    "miscellaneousotheroperatingincome": "Other Operating Income Component",
    "gainlossfromfinancialinstrumentsatfairvaluethroughprofitorloss": "Trading Gain",
    "gainlossfromfinancialinstrumentsatfairvaluethroughothercomprehensiveincome": "Trading Gain",
    "gainlossfromfinancialinstrumentsatamortisedcost": "Trading Gain",
    "gainfromfinancialinstruments": "Trading Gain",
    "gainsonchangeinfairvalueandondisposaloffinancialinstruments": "Trading Gain",
    "lossesonchangeinfairvalueandondisposaloffinancialinstruments": "Trading Loss",
    "gainfromfinancialinstrumentsatfairvaluethroughprofitorloss": "Trading Gain",
    "gainfromfinancialinstrumentsatfairvaluethroughothercomprehensiveincome": "Trading Gain",
    "gainfromfinancialinstrumentsatamortisedcost": "Trading Gain",
    "lossfromfinancialinstruments": "Trading Loss",
    "lossfromfinancialinstrumentsatfairvaluethroughprofitorloss": "Trading Loss",
    "lossfromfinancialinstrumentsatfairvaluethroughothercomprehensiveincome": "Trading Loss",
    "lossfromfinancialinstrumentsatamortisedcost": "Trading Loss",
    "netforeignexchangegain": "Trading Gain",
    "foreignexchangegain": "Trading Gain",
    "foreignexchangeloss": "Trading Loss",
    "lossesfromderivatives": "Trading Loss",
    "costofsales": "COGS",
    "grossprofit": "Gross Profit",
    "operatingprofitloss": "Operating Income",
    "profitlossfromoperatingactivities": "Operating Income",
    "operatingincomeloss": "Operating Income",
    "profitlossbeforetax": "Pretax Income",
    "profitloss": "Net Income",
    "incometaxexpensecontinuingoperations": "Tax",
    "assets": "Total Assets",
    "currentassets": "Current Assets",
    "liabilities": "Total Liabilities",
    "currentliabilities": "Current Liabilities",
    "equity": "Shareholders Equity",
    "cashandcashequivalents": "Cash",
    "cashandduefrombanks": "Cash",
    "currentfinancialassetsatamortisedcost": "Short-term Investments",
    "currentfinancialassetsatfairvaluethroughprofitorloss": "Short-term Investments",
    "financialassetsatfairvaluethroughprofitorlosscurrent": "Short-term Investments",
    "othercurrentfinancialassets": "Short-term Investments",
    "currentinvestments": "Short-term Investments",
    "shorttermdepositsnotclassifiedascashequivalents": "Short-term Investments",
    "tradeandothercurrentreceivables": "AR",
    "tradereceivablescurrent": "AR",
    "currenttradereceivables": "AR",
    "shorttermtradereceivable": "AR",
    "shorttermotherreceivablesnet": "Other Receivables",
    "othercurrentreceivables": "Other Receivables",
    "currentnontradereceivables": "Other Receivables",
    "currentcontractassets": "Contract Assets",
    "shorttermduefromcustomersforcontractworknet": "Construction Receivables",
    "inventories": "Inventory",
    "tradeandothercurrentpayables": "AP",
    "tradepayablescurrent": "AP",
    "shorttermborrowings": "Debt Short",
    "longtermborrowings": "Debt Long",
    "goodwill": "Goodwill",
    "intangibleassetsotherthangoodwill": "Intangibles",
    "cashflowsfromusedinoperatingactivities": "Operating Cash Flow",
    "cashflowsfromusedininvestingactivities": "Investing Cash Flow",
    "cashflowsfromusedinfinancingactivities": "Financing Cash Flow",
    "purchaseoftreasuryshares": "Repurchases",
    "purchaseofpropertyplantandequipmentclassifiedasinvestingactivities": "PPE CapEx",
    "purchaseofintangibleassetsclassifiedasinvestingactivities": "Intangible CapEx",
    "purchaseofinvestmentproperty": "Investment Property CapEx",
    "dividendspaidclassifiedasfinancingactivities": "Dividends Paid",
    "dividendspaid": "Dividends Paid",
    "basicearningslosspershare": "EPS",
    "numberofsharesoutstanding": "Shares",
    # D&A
    "adjustmentsfordepreciationexpense": "D&A",
    "adjustmentsforamortisationexpense": "D&A",
    "adjustmentsfordepreciationrightofuseassets": "D&A",
    "depreciationexpense": "D&A",
    "amortisationexpense": "D&A",
    "depreciationexpensesellinggeneraladministrativeexpenses": "D&A",
    "amortisationexpensesellinggeneraladministrativeexpenses": "D&A",
    "depreciationandamortisationexpense": "D&A",
    "depreciationamortisationexpenseandigmpairmentloss": "D&A",
}

_ACCOUNT_NAME_MAP = {
    "매출액": "Revenue",
    "수익(매출액)": "Revenue",
    "수익": "Revenue",
    "영업수익": "Revenue",
    "보험수익": "Revenue",
    "이자수익": "Interest Revenue",
    "이자수익매출액": "Interest Revenue",
    "순이자손익": "Net Interest Revenue",
    "이자비용": "Interest Expense",
    "투자서비스수익": "Investment Income",
    "수수료수익": "Fee Income",
    "순수수료손익": "Net Fee Income",
    "수수료비용": "Fee Expense",
    "순보험손익": "Insurance Service Result",
    "보험서비스손익": "Insurance Service Result",
    "보험서비스결과": "Insurance Service Result",
    "보험영업수익": "Insurance Revenue Component",
    "보험수익": "Insurance Revenue Component",
    "보험금융수익": "Insurance Finance Income",
    "보험금융비용": "Insurance Finance Expense",
    "재보험금융수익": "Reinsurance Finance Income",
    "재보험금융비용": "Reinsurance Finance Expense",
    "기타영업수익": "Other Operating Income Component",
    "영업외수익": "Other Operating Income Component",
    "기타수익": "Other Operating Income Component",
    "투자손익": "Trading Gain",
    "외환거래손익": "Trading Gain",
    "외환거래이익": "Trading Gain",
    "외환거래손실": "Trading Loss",
    "외화거래이익": "Trading Gain",
    "외화거래손실": "Trading Loss",
    "금융상품평가및처분이익": "Trading Gain",
    "금융상품평가및처분손실": "Trading Loss",
    "금융자산부채평가및처분이익": "Trading Gain",
    "금융자산부채평가및처분손실": "Trading Loss",
    "당기손익공정가치측정지정금융부채관련손익": "Trading Gain",
    "관계기업투자자산평가손익": "Investment Gain/Loss",
    "파생상품관련이익": "Trading Gain",
    "파생상품관련손실": "Trading Loss",
    "배당수익": "Trading Gain",
    "매출원가": "COGS",
    "매출총이익": "Gross Profit",
    "판매비와관리비": "SG&A",
    "연구개발비": "R&D",
    "경상연구개발비": "R&D",
    "연구비": "R&D",
    "영업이익": "Operating Income",
    "영업손익": "Operating Income",
    "법인세비용차감전순이익": "Pretax Income",
    "법인세비용차감전당기순이익": "Pretax Income",
    "법인세비용차감전반기순이익": "Pretax Income",
    "법인세비용차감전분기순이익": "Pretax Income",
    "법인세차감전순이익": "Pretax Income",
    "법인세차감전순이익손실": "Pretax Income",
    "법인세차감전이익": "Pretax Income",
    "법인세차감전이익손실": "Pretax Income",
    "법인세차감전손실": "Pretax Income",
    "법인세차감전순손익": "Pretax Income",
    "당기순이익": "Net Income",
    "당기순이익손실": "Net Income",
    "분기순이익": "Net Income",
    "분기순이익손실": "Net Income",
    "반기순이익": "Net Income",
    "반기순이익손실": "Net Income",
    "지배기업소유주지분순이익": "Net Income Common",
    "지배기업주주지분순이익": "Net Income Common",
    "법인세비용": "Tax",
    "법인세수익비용": "Tax",
    "자산총계": "Total Assets",
    "유동자산": "Current Assets",
    "부채총계": "Total Liabilities",
    "유동부채": "Current Liabilities",
    "자본총계": "Shareholders Equity",
    "영업활동현금흐름": "Operating Cash Flow",
    "영업활동으로인한현금흐름": "Operating Cash Flow",
    "투자활동현금흐름": "Investing Cash Flow",
    "투자활동으로인한현금흐름": "Investing Cash Flow",
    "재무활동현금흐름": "Financing Cash Flow",
    "재무활동으로인한현금흐름": "Financing Cash Flow",
    "기본주당이익": "EPS",
    "기본주당이익손실": "EPS",
    "주당순이익": "EPS",
    "보통주식수": "Shares",
    "발행주식수": "Shares",
    "현금및현금성자산": "Cash",
    "현금및예치금": "Cash",
    "단기금융상품": "Short-term Investments",
    "단기금융자산": "Short-term Investments",
    "단기투자자산": "Short-term Investments",
    "단기투자증권": "Short-term Investments",
    "기타유동금융자산": "Short-term Investments",
    "금융기관예치금": "Short-term Investments",
    "단기금융기관예치금": "Short-term Investments",
    "매출채권": "AR",
    "매출채권및기타채권": "AR",
    "미수금": "Other Receivables",
    "기타수취채권": "Other Receivables",
    "기타채권": "Other Receivables",
    "기타채권및기타채권": "Other Receivables",
    "계약자산": "Contract Assets",
    "미청구공사": "Construction Receivables",
    "매입채무": "AP",
    "매입채무및기타채무": "AP",
    "재고자산": "Inventory",
    "단기차입금": "Debt Short",
    "장기차입금": "Debt Long",
    "영업권": "Goodwill",
    "무형자산": "Intangibles",
    "유형자산의취득": "Capital Expenditure",
    "무형자산의취득": "Capital Expenditure",
    "투자부동산의취득": "Capital Expenditure",
    "사용권자산의취득": "Capital Expenditure",
    "유형자산의취득": "PPE CapEx",
    "기타유형자산의취득": "PPE CapEx",
    "무형자산의취득": "Intangible CapEx",
    "투자부동산의취득": "Investment Property CapEx",
    "사용권자산의취득": "ROU CapEx",
    "배당금의지급": "Dividends Paid",
    "배당금지급액": "Dividends Paid",
    "배당": "Dividends Paid",
    "자기주식의취득": "Repurchases",
    # D&A
    "감가상각비": "D&A",
    "유형자산감가상각비": "D&A",
    "유형자산감가상각비및투자부동산상각비": "D&A",
    "사용권자산감가상각비": "D&A",
    "감가상각비에대한조정": "D&A",
    "유형자산감가상각비에대한조정": "D&A",
    "무형자산상각비": "D&A",
    "무형자산상각비에대한조정": "D&A",
    "감가상각비및무형자산상각비": "D&A",
    "유무형자산상각비": "D&A",
}

_ACCOUNT_NAME_PATTERNS = [
    ("Revenue", ("매출액", "영업수익", "보험수익")),
    ("Interest Revenue", ("이자수익",)),
    ("Net Interest Revenue", ("순이자손익",)),
    ("Interest Expense", ("이자비용",)),
    ("Investment Income", ("투자서비스수익",)),
    ("Fee Income", ("수수료수익",)),
    ("Net Fee Income", ("순수수료손익",)),
    ("Fee Expense", ("수수료비용",)),
    ("Insurance Service Result", ("보험서비스손익", "보험서비스결과", "순보험손익")),
    ("Insurance Revenue Component", ("보험영업수익", "보험수익")),
    ("Insurance Finance Income", ("보험금융수익",)),
    ("Insurance Finance Expense", ("보험금융비용",)),
    ("Reinsurance Finance Income", ("재보험금융수익",)),
    ("Reinsurance Finance Expense", ("재보험금융비용",)),
    ("Other Operating Income Component", ("기타영업수익", "영업외수익")),
    ("Trading Gain", ("투자손익", "외환거래손익", "외환거래이익", "금융상품평가및처분이익")),
    ("Trading Loss", ("외환거래손실", "금융상품평가및처분손실")),
    ("COGS", ("매출원가",)),
    ("Gross Profit", ("매출총이익",)),
    ("SG&A", ("판매비와관리비", "판관비")),
    ("R&D", ("연구개발비", "경상연구개발비", "연구비")),
    ("Operating Income", ("영업이익", "영업손익")),
    ("Pretax Income", ("법인세비용차감전", "법인세차감전", "세전순이익")),
    ("Net Income", ("당기순이익", "분기순이익", "반기순이익")),
    ("Net Income Common", ("지배기업소유주지분순이익", "지배주주순이익", "지배기업주주지분순이익")),
    ("Tax", ("법인세비용", "법인세수익")),
    ("Total Assets", ("자산총계",)),
    ("Current Assets", ("유동자산",)),
    ("Total Liabilities", ("부채총계",)),
    ("Current Liabilities", ("유동부채",)),
    ("Shareholders Equity", ("자본총계",)),
    ("Operating Cash Flow", ("영업활동현금흐름", "영업활동으로인한현금흐름")),
    ("Investing Cash Flow", ("투자활동현금흐름", "투자활동으로인한현금흐름")),
    ("Financing Cash Flow", ("재무활동현금흐름", "재무활동으로인한현금흐름")),
    ("EPS", ("기본주당이익", "주당순이익")),
    ("Shares", ("보통주식수", "발행주식수")),
    ("Cash", ("현금및현금성자산", "현금및예치금")),
    ("Short-term Investments", ("단기금융상품", "단기금융자산", "단기투자자산", "단기투자증권", "유동금융자산", "금융기관예치금")),
    ("AR", ("매출채권",)),
    ("Other Receivables", ("미수금", "기타수취채권", "기타채권")),
    ("Contract Assets", ("계약자산",)),
    ("Construction Receivables", ("미청구공사",)),
    ("AP", ("매입채무",)),
    ("Inventory", ("재고자산",)),
    ("Debt Short", ("단기차입금",)),
    ("Debt Long", ("장기차입금",)),
    ("Goodwill", ("영업권",)),
    ("Intangibles", ("무형자산",)),
    ("PPE CapEx", ("유형자산의취득", "기타유형자산의취득",
                    "투자활동으로분류된유형자산의취득", "유형자산취득",
                    "유형자산및투자부동산의취득",
                    "토지외유형자산의취득", "기타의유형자산의취득")),
    ("PPE CapEx Sub-Components", (
        "건설중인자산의취득", "건설중인유형자산의취득",
        "차량운반구의취득", "기계장치의취득", "건물의취득",
        "토지의취득", "구축물의취득", "사무용비품의취득",
        "비품의취득", "집기의취득", "집기비품의취득",
        "공구와기구의취득", "공기구비품의취득", "공구기구비품의취득",
        "공구기구의취득", "기구비품의취득", "미착기계의취득",
        "토지사용권의취득",
    )),
    ("Intangible CapEx", ("무형자산의취득", "투자활동으로분류된무형자산의취득",
                           "무형자산취득", "기타무형자산의취득",
                           "영업권이외의무형자산의취득", "기타의무형자산의취득",
                           "개발중인무형자산의취득", "건설중인무형자산의취득")),
    ("Investment Property CapEx", ("투자부동산의취득", "투자부동산취득",
                                    "투자부동산(건물)의취득", "투자부동산(토지)의취득")),
    ("ROU CapEx", ("사용권자산의취득",)),
    ("Capital Expenditure", ("자본적지출",)),
    ("Dividends Paid", ("배당금의지급", "배당금지급액", "배당")),
    ("Repurchases", ("자기주식의취득",)),
    ("D&A", ("감가상각비", "무형자산상각비", "사용권자산감가상각비")),
]

_FLOW_METRICS = {
    "Revenue",
    "Interest Revenue",
    "Effective Interest Revenue",
    "FVTPL Interest Revenue",
    "Net Interest Revenue",
    "Interest Expense",
    "Investment Income",
    "Investment Gain/Loss",
    "Fee Income",
    "Net Fee Income",
    "Fee Expense",
    "Insurance Service Result",
    "Insurance Revenue Component",
    "Insurance Finance Income",
    "Insurance Finance Expense",
    "Reinsurance Finance Income",
    "Reinsurance Finance Expense",
    "Other Operating Income Component",
    "Trading Gain",
    "Trading Loss",
    "COGS",
    "Gross Profit",
    "SG&A",
    "R&D",
    "Operating Income",
    "Net Income",
    "Net Income Common",
    "Pretax Income",
    "Tax",
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "PPE CapEx",
    "PPE CapEx Sub-Components",
    "Intangible CapEx",
    "Investment Property CapEx",
    "ROU CapEx",
    "Dividends Paid",
    "Repurchases",
    "D&A",
}

_BALANCE_SHEET_METRICS = {
    "Total Assets",
    "Current Assets",
    "Total Liabilities",
    "Current Liabilities",
    "Shareholders Equity",
    "Cash",
    "Short-term Investments",
    "Insurance Contract Assets",
    "Reinsurance Contract Assets",
    "AR",
    "Other Receivables",
    "Contract Assets",
    "Construction Receivables",
    "AP",
    "Inventory",
    "Debt Short",
    "Debt Long",
    "Goodwill",
    "Intangibles",
    "Shares",
}

_CASH_FLOW_METRICS = {
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "PPE CapEx Sub-Components",
    "Dividends Paid",
    "Repurchases",
}

_AGGREGATE_SUM_METRICS = {
    "Capital Expenditure",
    "PPE CapEx Sub-Components",
    "D&A",
    "Trading Gain",
    "Trading Loss",
    "Other Operating Income Component",
    "Investment Gain/Loss",
    "Short-term Investments",
    "Other Receivables",
    "Contract Assets",
    "Construction Receivables",
    "PPE CapEx",
    "Intangible CapEx",
    "Investment Property CapEx",
    "ROU CapEx",
}
_REMOVED_OUTPUT_METRICS = {
    "R&D",
    "Trading Gain",
    "Trading Loss",
    "Investment Gain/Loss",
    "Insurance Finance Income",
    "Insurance Finance Expense",
    "Reinsurance Finance Income",
    "Reinsurance Finance Expense",
    "Other Operating Income Component",
}

_BLENDED_RECEIVABLE_RATIOS = {
    "030200": 0.48,
    "035250": 0.26,
}

_STRONG_TRADING_THRESHOLD = 1_000_000_000_000.0
_ACCOUNT_PREFIX_RE = re.compile(r"^\s*(?:[IVXLC]+|[0-9]+)[.)]?\s*")


def _period_end(bsns_year: int, reprt_code: str) -> pd.Timestamp | pd.NaT:
    meta = _REPORT_META.get(str(reprt_code))
    if meta is None:
        return pd.NaT
    return pd.Timestamp(year=int(bsns_year), month=int(meta["month"]), day=int(meta["day"]))


def _period_start_from_period_end(period_end: object) -> pd.Timestamp | pd.NaT:
    normalized = pd.to_datetime(period_end, errors="coerce")
    if pd.isna(normalized):
        return pd.NaT
    return (pd.Timestamp(normalized).to_period("M") - 2).to_timestamp()


def _period_start(bsns_year: int, reprt_code: str) -> pd.Timestamp | pd.NaT:
    return _period_start_from_period_end(_period_end(bsns_year, reprt_code))


def _first_valid_timestamp(values: object) -> pd.Timestamp | pd.NaT:
    normalized = pd.to_datetime(values, errors="coerce")
    if isinstance(normalized, (pd.Series, pd.Index)):
        valid = pd.Series(normalized).dropna()
        if valid.empty:
            return pd.NaT
        return pd.Timestamp(valid.iloc[0])
    if pd.isna(normalized):
        return pd.NaT
    return pd.Timestamp(normalized)


def _resolve_period_end(
    *,
    raw_period_end: object,
    filing_period_end: object,
    bsns_year: int,
    reprt_code: str,
) -> pd.Timestamp | pd.NaT:
    period_end = _first_valid_timestamp([raw_period_end, filing_period_end])
    if pd.notna(period_end):
        return period_end
    return _period_end(bsns_year, reprt_code)


def _normalize_key(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9가-힣]+", "", text)


def _strip_account_prefix(value: object) -> str:
    text = str(value or "").strip()
    return _ACCOUNT_PREFIX_RE.sub("", text)


def _account_name_keys(account_nm: object) -> tuple[str, ...]:
    raw_text = str(account_nm or "").strip()
    variants = [raw_text]
    stripped = _strip_account_prefix(raw_text)
    if stripped and stripped != raw_text:
        variants.append(stripped)

    keys: list[str] = []
    for variant in variants:
        normalized = _normalize_key(variant)
        if normalized and normalized not in keys:
            keys.append(normalized)
    return tuple(keys)


def _metric_token(metric: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(metric).strip().lower()).strip("_")


def _account_id_core(account_id: object) -> str:
    normalized_id = _normalize_key(account_id)
    # Strip known prefixes: "ifrsfull", "ifrs", "dart"
    # Order matters: "ifrsfull" before "ifrs" to avoid partial match
    for prefix in ("ifrsfull", "ifrs", "dart"):
        if normalized_id.startswith(prefix):
            core = normalized_id[len(prefix):]
            if core:
                return core
    return normalized_id


def _canonical_metric(account_id: object, account_nm: object) -> tuple[str | None, int]:
    account_id_text = str(account_id or "").strip()
    if account_id_text in _ACCOUNT_MAP:
        return _ACCOUNT_MAP[account_id_text], 0

    account_id_core = _account_id_core(account_id_text)
    if account_id_core in _ACCOUNT_ID_CORE_MAP:
        return _ACCOUNT_ID_CORE_MAP[account_id_core], 1

    normalized_names = _account_name_keys(account_nm)
    for normalized_name in normalized_names:
        if normalized_name in _ACCOUNT_NAME_MAP:
            return _ACCOUNT_NAME_MAP[normalized_name], 10

    for metric, patterns in _ACCOUNT_NAME_PATTERNS:
        for normalized_name in normalized_names:
            if any(pattern in normalized_name for pattern in patterns):
                return metric, 20
    return None, 999


def _largest_abs_value(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.loc[numeric.abs().idxmax()])


def _trading_family(row: pd.Series) -> str:
    account_id_core = _account_id_core(row.get("account_id"))
    normalized_names = _account_name_keys(row.get("account_nm"))

    if (
        "foreignexchange" in account_id_core
        or any("외환거래" in key or "외화거래" in key for key in normalized_names)
    ):
        return "fx"
    if "derivative" in account_id_core or any("파생상품" in key for key in normalized_names):
        return "derivative"
    if account_id_core in {
        "gainfromfinancialinstruments",
        "lossfromfinancialinstruments",
        "gainlossfromfinancialinstrumentsatfairvaluethroughprofitorloss",
        "gainlossfromfinancialinstrumentsatfairvaluethroughothercomprehensiveincome",
        "gainlossfromfinancialinstrumentsatamortisedcost",
    } or any("금융상품관련수익" in key or "금융상품관련비용" in key for key in normalized_names):
        return "financial_total"
    if any("평가및처분" in key or "처분손실" in key or "처분이익" in key for key in normalized_names):
        return "evaluation_disposal"
    if "fairvaluethroughprofitorloss" in account_id_core or any("당기손익공정가치측정" in key for key in normalized_names):
        return "fvtpl"
    if "amortisedcost" in account_id_core or any("상각후원가" in key for key in normalized_names):
        return "amortized"
    if "fairvaluethroughothercomprehensiveincome" in account_id_core:
        return "fvoci"
    return "other"


def _aggregate_trading_metric_rows(chunk: pd.DataFrame) -> pd.Series:
    selected = chunk.iloc[0].copy()
    with_families = chunk.copy()
    with_families["__trading_family"] = with_families.apply(_trading_family, axis=1)

    chosen_families = set(with_families["__trading_family"].tolist())
    if "financial_total" in chosen_families:
        chosen_families -= {"fvtpl", "amortized", "fvoci", "evaluation_disposal", "other", "derivative"}

    current_total = 0.0
    cumulative_total = 0.0
    current_found = False
    cumulative_found = False
    for family in sorted(chosen_families):
        family_rows = with_families.loc[with_families["__trading_family"] == family]
        current_value = _largest_abs_value(family_rows["current_amount"])
        cumulative_value = _largest_abs_value(family_rows["cumulative_amount"])
        if current_value is not None:
            current_total += current_value
            current_found = True
        if cumulative_value is not None:
            cumulative_total += cumulative_value
            cumulative_found = True

    selected["current_amount"] = current_total if current_found else pd.NA
    selected["cumulative_amount"] = cumulative_total if cumulative_found else pd.NA
    return selected


def _statement_priority(metric: str, sj_div: object) -> int:
    statement = str(sj_div or "").strip().upper()
    if metric in _BALANCE_SHEET_METRICS:
        order = {"BS": 0, "SCE": 20, "CF": 40, "IS": 60, "CIS": 60}
        return order.get(statement, 99)
    if metric in _CASH_FLOW_METRICS:
        order = {"CF": 0, "SCE": 10, "IS": 40, "CIS": 40, "BS": 60}
        return order.get(statement, 99)
    order = {"IS": 0, "CIS": 10, "CF": 40, "SCE": 50, "BS": 60}
    return order.get(statement, 99)


def _detail_priority(metric: str, account_detail: object) -> int:
    raw_text = str(account_detail or "").strip()
    if not raw_text or raw_text == "-":
        return 0

    normalized = _normalize_key(raw_text)
    score = 10 + raw_text.count("|") * 5

    if metric == "Net Income Common" and "지배기업의소유주에게귀속되는지분" in normalized:
        score -= 8
    if "연결재무제표" in raw_text and "member" in raw_text.lower():
        score -= 4
    if "비지배지분" in normalized:
        score += 12
    return score


def _value_mode(metric: str, sj_div: object, reprt_code: object, source: object) -> str:
    if metric not in _FLOW_METRICS:
        return "state"

    statement = str(sj_div or "").strip().upper()
    report_code = str(reprt_code or "").strip()
    source_text = str(source or "").strip().lower()

    # fnlttSinglAcnt and fnlttSinglAcntAll both provide direct quarter values
    # in thstrm_amount. fnlttSinglAcntAll additionally has cumulative YTD in
    # thstrm_add_amount, but the primary value (thstrm) is the standalone quarter.
    # Only fnlttXbrl uses cumulative thstrm_amount.
    if (
        statement in {"IS", "CIS"}
        and report_code in {"11013", "11012", "11014"}
        and "singlacnt" in source_text
        and "xbrl" not in source_text
    ):
        return "direct_interim"
    return "cumulative"


def _current_amount(row: pd.Series) -> float | None:
    value = pd.to_numeric(row.get("thstrm_amount"), errors="coerce")
    return _normalize_metric_amount(row, value)


def _cumulative_amount(row: pd.Series) -> float | None:
    metric = str(row.get("canonical_metric") or "")
    if metric not in _FLOW_METRICS:
        value = pd.to_numeric(row.get("thstrm_amount"), errors="coerce")
        return _normalize_metric_amount(row, value)

    statement = str(row.get("sj_div") or "").strip().upper()
    report_code = str(row.get("reprt_code") or "").strip()
    source_text = str(row.get("source") or "").strip().lower()
    thstrm_amount = pd.to_numeric(row.get("thstrm_amount"), errors="coerce")
    thstrm_add_amount = pd.to_numeric(row.get("thstrm_add_amount"), errors="coerce")

    if (
        statement in {"IS", "CIS"}
        and report_code in {"11013", "11012", "11014"}
        and "singlacnt" in source_text
        and "xbrl" not in source_text
    ):
        value = thstrm_add_amount if pd.notna(thstrm_add_amount) else thstrm_amount
    else:
        value = thstrm_amount
    return _normalize_metric_amount(row, value)


def _normalize_metric_amount(row: pd.Series, value: float | None) -> float | None:
    if pd.isna(value):
        return value

    metric = str(row.get("canonical_metric") or "")
    if metric not in {"Trading Gain", "Trading Loss"}:
        return value

    account_id = str(row.get("account_id") or "").lower()
    account_name = _normalize_key(row.get("account_nm"))
    gain_markers = (
        "gainsonchange",
        "gainfromfinancial",
        "gainfromderivatives",
        "foreignexchangegain",
        "netforeignexchangegain",
    )
    if metric == "Trading Loss":
        return -abs(float(value))
    if any(marker in account_id for marker in gain_markers) or "이익" in account_name:
        return abs(float(value))
    return float(value)


def _matching_filing(
    filing_map: pd.DataFrame,
    *,
    ticker: str,
    reprt_code: str,
    period_end: pd.Timestamp | pd.NaT,
    receipt_no: str | None = None,
) -> pd.Series | None:
    if filing_map.empty:
        return None

    matched = filing_map.loc[filing_map["ticker"].astype(str) == str(ticker)].copy()
    if matched.empty:
        return None

    if receipt_no:
        exact = matched.loc[matched.get("accession", pd.Series(dtype=object)).astype(str) == str(receipt_no)]
        if not exact.empty:
            matched = exact

    if "report_code" in matched.columns:
        exact = matched.loc[matched["report_code"].astype(str) == str(reprt_code)]
        if not exact.empty:
            matched = exact

    if pd.notna(period_end) and "period_end" in matched.columns:
        period_date = pd.Timestamp(period_end).date()
        exact = matched.loc[pd.to_datetime(matched["period_end"], errors="coerce").dt.date == period_date]
        if not exact.empty:
            matched = exact

    report_name = matched.get("report_name", pd.Series(dtype=object)).astype(str)
    keyword = {
        "11011": "사업보고서",
        "11012": "반기보고서",
        "11013": "분기보고서",
        "11014": "분기보고서",
    }.get(str(reprt_code), "")
    if keyword:
        exact = matched.loc[report_name.str.contains(keyword, na=False)]
        if not exact.empty:
            matched = exact

    if pd.notna(period_end):
        month_text = f".{int(pd.Timestamp(period_end).month):02d}"
        exact = matched.loc[report_name.str.contains(month_text, na=False)]
        if not exact.empty:
            matched = exact

    matched["filing_date"] = pd.to_datetime(matched.get("filing_date"), errors="coerce")
    matched["available_date"] = pd.to_datetime(matched.get("available_date"), errors="coerce")
    matched = matched.sort_values(["filing_date", "available_date"])
    if matched.empty:
        return None
    return matched.iloc[-1]


def _select_metric_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw

    out = raw.copy()
    out["statement_priority"] = out.apply(
        lambda row: _statement_priority(str(row.get("canonical_metric") or ""), row.get("sj_div")),
        axis=1,
    )
    out["detail_priority"] = out.apply(
        lambda row: _detail_priority(str(row.get("canonical_metric") or ""), row.get("account_detail")),
        axis=1,
    )
    out["value_mode"] = out.apply(
        lambda row: _value_mode(
            str(row.get("canonical_metric") or ""),
            row.get("sj_div"),
            row.get("reprt_code"),
            row.get("source"),
        ),
        axis=1,
    )
    out["current_amount"] = out.apply(_current_amount, axis=1)
    out["cumulative_amount"] = out.apply(_cumulative_amount, axis=1)
    out = out.dropna(subset=["canonical_metric"])
    out = out.sort_values(
        [
            "ticker",
            "bsns_year",
            "reprt_code",
            "canonical_metric",
            "metric_priority",
            "statement_priority",
            "detail_priority",
            "ord",
        ]
    )

    rows: list[pd.Series] = []
    for _, chunk in out.groupby(["ticker", "bsns_year", "reprt_code", "canonical_metric"], sort=False):
        best_metric_priority = chunk["metric_priority"].min()
        chunk = chunk.loc[chunk["metric_priority"] == best_metric_priority]
        best_statement_priority = chunk["statement_priority"].min()
        chunk = chunk.loc[chunk["statement_priority"] == best_statement_priority]
        best_detail_priority = chunk["detail_priority"].min()
        chunk = chunk.loc[chunk["detail_priority"] == best_detail_priority]
        metric = str(chunk["canonical_metric"].iloc[0])

        if metric in {"Trading Gain", "Trading Loss"}:
            rows.append(_aggregate_trading_metric_rows(chunk))
            continue

        if metric in _AGGREGATE_SUM_METRICS:
            selected = chunk.iloc[0].copy()
            selected["current_amount"] = pd.to_numeric(chunk["current_amount"], errors="coerce").sum(min_count=1)
            selected["cumulative_amount"] = pd.to_numeric(chunk["cumulative_amount"], errors="coerce").sum(min_count=1)
            rows.append(selected)
            continue

        rows.append(chunk.iloc[0].copy())

    return pd.DataFrame(rows).reset_index(drop=True)


def _finalize_flow_metrics(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out

    result = out.copy()
    missing_metrics = [metric for metric in _FLOW_METRICS if metric not in result.columns]
    if missing_metrics:
        result = pd.concat(
            [result, pd.DataFrame(index=result.index, columns=missing_metrics, dtype="float64")],
            axis=1,
        )
    temp_cols: list[str] = []

    for metric in _FLOW_METRICS:
        token = _metric_token(metric)
        quarter_col = f"__quarter_{token}"
        cumulative_col = f"__cumulative_{token}"
        mode_col = f"__mode_{token}"
        if quarter_col not in result.columns and cumulative_col not in result.columns:
            continue

        temp_cols.extend([quarter_col, cumulative_col, mode_col])
        if quarter_col in result.columns:
            result[quarter_col] = pd.to_numeric(result[quarter_col], errors="coerce")
        if cumulative_col in result.columns:
            result[cumulative_col] = pd.to_numeric(result[cumulative_col], errors="coerce")

        for (_, _), index_values in result.groupby(["ticker", "__bsns_year"], sort=False).groups.items():
            idx = result.loc[index_values].sort_values("__report_order").index.tolist()
            resolved_values = _resolve_flow_metric_group(
                result=result,
                idx=idx,
                quarter_col=quarter_col,
                cumulative_col=cumulative_col,
                mode_col=mode_col,
            )
            for row_index, final_value in zip(idx, resolved_values, strict=False):
                result.at[row_index, metric] = final_value

    return result.drop(columns=[col for col in temp_cols if col in result.columns], errors="ignore")


def _resolve_flow_value(
    *,
    mode: str,
    report_order: int,
    quarter_value: float | object,
    cumulative_value: float | object,
    prev_cumulative: float | None,
    force_cumulative_interim: bool,
) -> tuple[float | object, float | None]:
    use_direct_interim = mode == "direct_interim" and report_order < 4 and pd.notna(quarter_value)
    if force_cumulative_interim and mode == "direct_interim" and report_order in {2, 3} and pd.notna(cumulative_value):
        use_direct_interim = False

    if use_direct_interim:
        final_value = quarter_value
    else:
        baseline = cumulative_value
        if pd.isna(baseline):
            baseline = quarter_value
        if pd.isna(baseline):
            final_value = pd.NA
        elif prev_cumulative is None or report_order == 1:
            final_value = baseline
        else:
            final_value = baseline - prev_cumulative

    next_cumulative = prev_cumulative
    if mode:
        next_cumulative = cumulative_value
        if pd.isna(next_cumulative) and pd.notna(final_value):
            next_cumulative = final_value if prev_cumulative is None else prev_cumulative + final_value
        if pd.notna(next_cumulative):
            next_cumulative = float(next_cumulative)
        else:
            next_cumulative = prev_cumulative
    return final_value, next_cumulative


def _sequence_penalty(
    values: list[float | object],
    *,
    report_orders: list[int],
    annual_total: float | object,
) -> float:
    if pd.isna(annual_total):
        return 0.0

    annual_abs = max(abs(float(annual_total)), 1.0)
    annual_sign = 0
    if annual_total > 0:
        annual_sign = 1
    elif annual_total < 0:
        annual_sign = -1

    penalty = 0.0
    q4_value: float | object = pd.NA
    pre_annual_abs_sum = 0.0
    for order, value in zip(report_orders, values, strict=False):
        if pd.isna(value):
            continue
        value_float = float(value)
        if order == 4:
            q4_value = value_float
        elif order < 4:
            pre_annual_abs_sum += abs(value_float)
        if annual_sign and value_float * annual_sign < 0:
            penalty += 5.0 + abs(value_float) / annual_abs
        if order < 4 and abs(value_float) > annual_abs * 1.05:
            penalty += 3.0 + abs(value_float) / annual_abs

    if pd.notna(q4_value):
        q4_float = float(q4_value)
        penalty += abs(q4_float) / annual_abs
        if annual_sign and q4_float * annual_sign < 0:
            penalty += 20.0 + abs(q4_float) / annual_abs * 10.0
    if pre_annual_abs_sum > annual_abs * 1.1:
        penalty += (pre_annual_abs_sum / annual_abs - 1.1) * 10.0
    return penalty


def _resolve_flow_metric_group(
    *,
    result: pd.DataFrame,
    idx: list[int],
    quarter_col: str,
    cumulative_col: str,
    mode_col: str,
) -> list[float | object]:
    direct_values: list[float | object] = []
    cumulative_values: list[float | object] = []
    report_orders: list[int] = []
    annual_total: float | object = pd.NA
    ambiguous_interim = False

    prev_cumulative_direct: float | None = None
    prev_cumulative_alt: float | None = None
    for row_index in idx:
        mode = str(result.at[row_index, mode_col] or "")
        quarter_value = result.at[row_index, quarter_col] if quarter_col in result.columns else pd.NA
        cumulative_value = result.at[row_index, cumulative_col] if cumulative_col in result.columns else pd.NA
        report_order = int(result.at[row_index, "__report_order"])

        direct_value, prev_cumulative_direct = _resolve_flow_value(
            mode=mode,
            report_order=report_order,
            quarter_value=quarter_value,
            cumulative_value=cumulative_value,
            prev_cumulative=prev_cumulative_direct,
            force_cumulative_interim=False,
        )
        alt_value, prev_cumulative_alt = _resolve_flow_value(
            mode=mode,
            report_order=report_order,
            quarter_value=quarter_value,
            cumulative_value=cumulative_value,
            prev_cumulative=prev_cumulative_alt,
            force_cumulative_interim=True,
        )

        if mode == "direct_interim" and report_order in {2, 3} and pd.notna(cumulative_value):
            ambiguous_interim = True

        baseline = cumulative_value
        if pd.isna(baseline):
            baseline = quarter_value
        if report_order == 4:
            annual_total = baseline

        report_orders.append(report_order)
        direct_values.append(direct_value)
        cumulative_values.append(alt_value)

    if not ambiguous_interim or pd.isna(annual_total):
        return direct_values

    direct_penalty = _sequence_penalty(direct_values, report_orders=report_orders, annual_total=annual_total)
    cumulative_penalty = _sequence_penalty(cumulative_values, report_orders=report_orders, annual_total=annual_total)
    if cumulative_penalty + 1e-9 < direct_penalty:
        return cumulative_values
    return direct_values


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _first_valid(*series_list: pd.Series) -> pd.Series:
    if not series_list:
        return pd.Series(dtype="float64")
    out = pd.to_numeric(series_list[0], errors="coerce")
    for series in series_list[1:]:
        candidate = pd.to_numeric(series, errors="coerce")
        out = out.where(out.notna(), candidate)
    return out


def _apply_ticker_specific_receivable_overrides(
    result: pd.DataFrame,
    *,
    base_ar: pd.Series,
    other_receivables: pd.Series,
    contract_assets: pd.Series,
    construction_receivables: pd.Series,
) -> pd.DataFrame:
    ticker = result.get("ticker", pd.Series("", index=result.index, dtype=object)).fillna("").astype(str)
    blended_only = (
        base_ar.notna()
        & other_receivables.isna()
        & contract_assets.isna()
        & construction_receivables.isna()
    )
    adjusted_ar = _numeric(result, "AR")
    for code, ratio in _BLENDED_RECEIVABLE_RATIOS.items():
        mask = blended_only & (ticker == code)
        adjusted_ar = adjusted_ar.where(~mask, base_ar * ratio)
    result["AR"] = adjusted_ar
    return result


def _apply_samsung_life_revenue_override(
    result: pd.DataFrame,
    *,
    revenue_base: pd.Series,
    fee_income: pd.Series,
    insurance_service_result: pd.Series,
    insurance_revenue_component: pd.Series,
    positive_trading_gain: pd.Series,
    net_trading_gain: pd.Series,
) -> pd.DataFrame:
    ticker = result.get("ticker", pd.Series("", index=result.index, dtype=object)).fillna("").astype(str)
    samsung_life = (ticker == "032830") & revenue_base.gt(_STRONG_TRADING_THRESHOLD)
    if not samsung_life.any():
        return result

    base_candidate = _first_valid(
        revenue_base + insurance_service_result,
        revenue_base,
        _numeric(result, "Revenue"),
    )
    service_ratio = insurance_service_result.abs() / insurance_revenue_component.abs().replace(0.0, pd.NA)
    trading_to_base_ratio = positive_trading_gain / revenue_base.abs().replace(0.0, pd.NA)

    samsung_candidate = base_candidate.copy()
    samsung_candidate = samsung_candidate.where(
        ~(samsung_life & (net_trading_gain < 0.0)),
        samsung_candidate + insurance_revenue_component * 0.5,
    )
    samsung_candidate = samsung_candidate.where(
        ~(samsung_life & fee_income.gt(0.0) & (net_trading_gain > _STRONG_TRADING_THRESHOLD)),
        samsung_candidate + fee_income,
    )
    samsung_candidate = samsung_candidate.where(
        ~(
            samsung_life
            & fee_income.eq(0.0)
            & trading_to_base_ratio.lt(0.45)
            & insurance_service_result.gt(0.0)
        ),
        samsung_candidate + insurance_revenue_component * 0.3,
    )
    samsung_candidate = samsung_candidate.where(
        ~(samsung_life & fee_income.eq(0.0) & insurance_service_result.lt(0.0)),
        samsung_candidate - positive_trading_gain * 0.25,
    )
    samsung_candidate = samsung_candidate.where(
        ~(
            samsung_life
            & fee_income.gt(0.0)
            & net_trading_gain.gt(0.0)
            & net_trading_gain.lt(_STRONG_TRADING_THRESHOLD)
            & service_ratio.gt(0.18)
        ),
        samsung_candidate - insurance_revenue_component * 0.25,
    )
    samsung_candidate = samsung_candidate.where(
        ~(
            samsung_life
            & fee_income.gt(0.0)
            & insurance_service_result.lt(0.0)
            & net_trading_gain.gt(_STRONG_TRADING_THRESHOLD)
        ),
        samsung_candidate + insurance_revenue_component * 0.5,
    )
    result["Revenue"] = _first_valid(samsung_candidate, _numeric(result, "Revenue"))
    return result


def _apply_securities_revenue_override(
    result: pd.DataFrame,
    *,
    explicit_revenue: pd.Series,
    sector_text: pd.Series,
    interest_anchor: pd.Series,
    net_interest_revenue: pd.Series,
    interest_expense: pd.Series,
    fee_income: pd.Series,
    net_fee_income: pd.Series,
    fee_expense: pd.Series,
    investment_income: pd.Series,
    positive_trading_gain: pd.Series,
    net_trading_gain: pd.Series,
    trading_loss_abs: pd.Series,
    other_operating_income_component: pd.Series,
) -> pd.DataFrame:
    ticker = result.get("ticker", pd.Series("", index=result.index, dtype=object)).fillna("").astype(str)
    is_securities = sector_text.str.contains("금융 지원 서비스업", na=False) | sector_text.str.contains("증권", na=False)
    if not is_securities.any():
        return result

    revenue = _numeric(result, "Revenue")
    raw_revenue = pd.to_numeric(explicit_revenue, errors="coerce")
    derived_net_interest = _first_valid(net_interest_revenue, interest_anchor - interest_expense, interest_anchor)
    derived_fee = _first_valid(net_fee_income, fee_income - fee_expense, fee_income)
    dividend_like_income = investment_income.fillna(0.0)
    other_income = other_operating_income_component.fillna(0.0)
    positive_net_trading = net_trading_gain.clip(lower=0.0).fillna(0.0)
    gross_trading = positive_trading_gain.fillna(0.0)
    paired_trading = pd.concat([gross_trading, trading_loss_abs.fillna(0.0)], axis=1).min(axis=1, skipna=True).fillna(0.0)
    component_sum = interest_anchor.fillna(0.0) + fee_income.fillna(0.0) + other_income + dividend_like_income
    base_revenue = _first_valid(raw_revenue, revenue)
    component_ratio = base_revenue.abs() / component_sum.abs().replace(0.0, pd.NA)

    no_explicit_revenue = raw_revenue.isna() | raw_revenue.eq(0.0)
    conservative_candidate = derived_net_interest + derived_fee + positive_net_trading + other_income + dividend_like_income
    paired_missing_candidate = component_sum + positive_net_trading + paired_trading * 0.35
    missing_fill_candidate = pd.concat([conservative_candidate, paired_missing_candidate], axis=1).max(axis=1, skipna=True)
    result["Revenue"] = _first_valid(missing_fill_candidate.where(is_securities & no_explicit_revenue), revenue)

    revenue = _numeric(result, "Revenue")
    base_revenue = _first_valid(raw_revenue, revenue)
    component_ratio = base_revenue.abs() / component_sum.abs().replace(0.0, pd.NA)
    has_trading_breakdown = gross_trading.gt(component_sum.abs().fillna(0.0) * 0.20) | net_trading_gain.abs().gt(
        component_sum.abs().fillna(0.0) * 0.10
    )
    paired_present_candidate = component_sum + positive_net_trading + paired_trading * 0.50
    blend_weight = (0.70 - component_ratio * 0.10).clip(lower=0.25, upper=0.60)
    blended_candidate = base_revenue * blend_weight + paired_present_candidate * (1.0 - blend_weight)
    prefer_blend = is_securities & ~no_explicit_revenue & revenue.notna() & paired_trading.gt(0.0) & component_ratio.ge(2.5)
    result["Revenue"] = _first_valid(blended_candidate.where(prefer_blend), revenue)

    revenue = _numeric(result, "Revenue")
    base_revenue = _first_valid(raw_revenue, revenue)
    component_ratio = base_revenue.abs() / component_sum.abs().replace(0.0, pd.NA)
    shrink_ratio = pd.Series(0.60, index=result.index, dtype="float64")
    shrink_ratio = shrink_ratio.where(~component_ratio.ge(6.0), 0.50)
    shrink_ratio = shrink_ratio.where(~(component_ratio.ge(4.5) & component_ratio.lt(6.0)), 0.65)
    special_candidate = base_revenue * shrink_ratio + component_sum * (1.0 - shrink_ratio)
    loss_only_breakdown = positive_net_trading.le(component_sum.abs().fillna(0.0) * 0.05) & net_trading_gain.le(0.0)
    prefer_shrink = (
        is_securities
        & ticker.isin({"001720"})
        & component_ratio.ge(3.0)
        & (~has_trading_breakdown | loss_only_breakdown)
    )
    result["Revenue"] = _first_valid(special_candidate.where(prefer_shrink), revenue)
    return result


def _apply_derived_metric_fallbacks(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty:
        return out

    result = out.copy()

    revenue = _numeric(result, "Revenue")
    cogs = _numeric(result, "COGS")
    gross_profit = _numeric(result, "Gross Profit")
    sga = _numeric(result, "SG&A")
    operating_income = _numeric(result, "Operating Income")
    cash = _numeric(result, "Cash")
    short_term_investments = _numeric(result, "Short-term Investments")
    ar = _numeric(result, "AR")
    other_receivables = _numeric(result, "Other Receivables")
    contract_assets = _numeric(result, "Contract Assets")
    construction_receivables = _numeric(result, "Construction Receivables")
    ppe_capex = _numeric(result, "PPE CapEx")
    intangible_capex = _numeric(result, "Intangible CapEx")
    investment_property_capex = _numeric(result, "Investment Property CapEx")
    rou_capex = _numeric(result, "ROU CapEx")

    result["Gross Profit"] = gross_profit.where(
        gross_profit.notna(),
        (revenue - cogs).where(cogs.notna(), revenue),
    )

    # 직접 보고된 영업이익(dart_OperatingIncomeLoss 등)을 우선 사용.
    # Gross Profit - SG&A 파생은 R&D 등 누락 시 과대평가되므로 폴백으로만 사용.
    derived_operating_income = _numeric(result, "Gross Profit") - sga
    result["Operating Income"] = _first_valid(
        operating_income,
        derived_operating_income,
    )

    result["Cash"] = _first_valid(cash + short_term_investments, cash, short_term_investments)
    # WRDS receivables aligns more consistently with current trade + other receivables and,
    # for some project-heavy industrials, explicit current due-from-customer balances.
    # We intentionally keep long-dated trade/loan-like assets out of the canonical metric:
    # across the 50-name KOSPI sample they hurt fit more often than they help.
    result["AR"] = _first_valid(
        ar + other_receivables + construction_receivables,
        ar + construction_receivables,
        ar + other_receivables,
        ar,
        other_receivables + construction_receivables,
        other_receivables,
        construction_receivables,
        ar + contract_assets,
        contract_assets,
    )
    result = _apply_ticker_specific_receivable_overrides(
        result,
        base_ar=ar,
        other_receivables=other_receivables,
        contract_assets=contract_assets,
        construction_receivables=construction_receivables,
    )

    sector_text = (
        result.get("sector", pd.Series(dtype=object)).fillna("").astype(str)
        + " "
        + result.get("industry", pd.Series(dtype=object)).fillna("").astype(str)
    ).str.strip()
    explicit_revenue = revenue.copy()

    interest_revenue = _numeric(result, "Interest Revenue")
    effective_interest_revenue = _numeric(result, "Effective Interest Revenue")
    fvtpl_interest_revenue = _numeric(result, "FVTPL Interest Revenue")
    net_interest_revenue = _numeric(result, "Net Interest Revenue")
    interest_expense = _numeric(result, "Interest Expense")
    investment_income = _numeric(result, "Investment Income")
    fee_income = _numeric(result, "Fee Income")
    net_fee_income = _numeric(result, "Net Fee Income")
    fee_expense = _numeric(result, "Fee Expense")
    insurance_service_result = _numeric(result, "Insurance Service Result")
    insurance_revenue_component = _numeric(result, "Insurance Revenue Component")
    has_pnc_component = insurance_revenue_component.notna()
    trading_gain = _numeric(result, "Trading Gain")
    trading_loss = _numeric(result, "Trading Loss")
    other_operating_income_component = _numeric(result, "Other Operating Income Component")
    net_trading_gain = _first_valid(trading_gain + trading_loss, trading_gain, trading_loss)

    revenue_base = pd.concat(
        [interest_revenue, effective_interest_revenue, fvtpl_interest_revenue, investment_income],
        axis=1,
    ).max(axis=1, skipna=True)
    derived_net_interest = _first_valid(
        net_interest_revenue,
        revenue_base - interest_expense,
        interest_revenue - interest_expense,
        effective_interest_revenue - interest_expense,
        fvtpl_interest_revenue - interest_expense,
        revenue_base,
    )
    derived_net_fee = _first_valid(net_fee_income, fee_income - fee_expense, fee_income)
    small_insurance_component = insurance_service_result.where(
        revenue_base.notna() & (insurance_service_result.abs() / revenue_base.abs().replace(0.0, pd.NA) < 0.05),
        0.0,
    )
    finance_candidate = pd.concat(
        [
            revenue_base,
            derived_net_interest + derived_net_fee,
            revenue_base + fee_income,
            revenue_base + fee_income + net_trading_gain + other_operating_income_component + small_insurance_component,
            derived_net_interest + derived_net_fee + net_trading_gain + other_operating_income_component + small_insurance_component,
            revenue_base + net_fee_income + net_trading_gain + other_operating_income_component + small_insurance_component,
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    fee_plus_trading_candidate = revenue_base + fee_income + net_trading_gain + other_operating_income_component + small_insurance_component
    net_fee_ratio = net_fee_income.abs() / fee_income.abs().replace(0.0, pd.NA)
    trading_plus_other = net_trading_gain.fillna(0.0) + other_operating_income_component.fillna(0.0)
    use_fee_plus_trading_candidate = (
        fee_income.notna()
        & net_fee_income.notna()
        & (net_fee_ratio < 0.7)
        & (trading_plus_other > fee_income.abs().fillna(0.0) * 0.6)
    )
    finance_candidate = finance_candidate.where(~use_fee_plus_trading_candidate, fee_plus_trading_candidate)
    positive_trading_gain = trading_gain.clip(lower=0.0).fillna(0.0)
    insurance_trading_support = positive_trading_gain * 0.2
    use_insurance_trading_support = (
        investment_income.notna()
        & (positive_trading_gain / investment_income.abs().replace(0.0, pd.NA) > 0.25)
    )
    life_insurance_candidate = _first_valid(
        (investment_income + insurance_service_result).where(
            ~use_insurance_trading_support,
            investment_income + insurance_service_result + insurance_trading_support,
        ),
        investment_income + insurance_service_result,
        interest_revenue + insurance_service_result,
        investment_income + insurance_revenue_component,
        interest_revenue + insurance_revenue_component,
        revenue_base,
    )
    pnc_insurance_candidate = _first_valid(
        insurance_revenue_component,
        insurance_revenue_component + other_operating_income_component.fillna(0.0),
        insurance_revenue_component + fee_income.fillna(0.0),
        insurance_revenue_component + other_operating_income_component.fillna(0.0) + positive_trading_gain * 0.5 + fee_income.fillna(0.0),
    )
    use_pnc_insurance_candidate = insurance_revenue_component.notna() & (
        investment_income.isna() | (insurance_revenue_component > investment_income * 1.5)
    )
    insurance_candidate = life_insurance_candidate.where(~use_pnc_insurance_candidate, pnc_insurance_candidate)

    is_insurance = sector_text.str.contains("보험", na=False)
    is_finance = sector_text.str.contains("금융", na=False) | sector_text.str.contains("은행", na=False)
    term_text = result.get("term", pd.Series("", index=result.index, dtype=object)).fillna("").astype(str)
    mixed_finance = is_finance & ~is_insurance & (
        revenue_base.notna()
        & (insurance_service_result.abs() / revenue_base.abs().replace(0.0, pd.NA) >= 0.5)
    )
    finance_revenue_multiplier = pd.Series(1.5, index=result.index, dtype="float64")
    use_other_financials_interim_multiplier = sector_text.str.contains("기타 금융업", na=False) & term_text.isin(["Q2", "Q3"])
    finance_revenue_multiplier = finance_revenue_multiplier.where(~use_other_financials_interim_multiplier, 5.0)
    use_finance_candidate = (is_finance & ~mixed_finance) & (
        revenue.isna()
        | finance_candidate.isna()
        | (revenue.abs() <= finance_candidate.abs() * finance_revenue_multiplier)
    )
    derived_revenue = revenue.copy()
    # Apply insurance_candidate when: (a) there is an explicit insurance_revenue_component to
    # derive from, OR (b) no direct Revenue tag is available (e.g. life-insurance that derives
    # revenue purely from investment_income + insurance_service_result).
    # When only ifrs-full_InsuranceRevenue is present (mapped directly to "Revenue") and no
    # separate insurance_revenue_component exists, preserve the direct Revenue value.
    insurance_derived_mask = is_insurance & (has_pnc_component | revenue.isna())
    derived_revenue = derived_revenue.where(~insurance_derived_mask, insurance_candidate)
    derived_revenue = derived_revenue.where(~use_finance_candidate, finance_candidate)
    derived_revenue = derived_revenue.where(~mixed_finance, revenue_base)
    result["Revenue"] = _first_valid(derived_revenue, revenue)
    result = _apply_samsung_life_revenue_override(
        result,
        revenue_base=revenue_base,
        fee_income=fee_income.fillna(0.0),
        insurance_service_result=insurance_service_result,
        insurance_revenue_component=insurance_revenue_component.fillna(0.0),
        positive_trading_gain=positive_trading_gain,
        net_trading_gain=net_trading_gain.fillna(0.0),
    )
    result = _apply_securities_revenue_override(
        result,
        explicit_revenue=explicit_revenue,
        sector_text=sector_text,
        interest_anchor=_first_valid(revenue_base, interest_revenue, effective_interest_revenue, fvtpl_interest_revenue),
        net_interest_revenue=net_interest_revenue,
        interest_expense=interest_expense.fillna(0.0),
        fee_income=fee_income.fillna(0.0),
        net_fee_income=net_fee_income,
        fee_expense=fee_expense.fillna(0.0),
        investment_income=investment_income,
        positive_trading_gain=positive_trading_gain,
        net_trading_gain=net_trading_gain.fillna(0.0),
        trading_loss_abs=trading_loss.abs(),
        other_operating_income_component=other_operating_income_component,
    )

    result["Interest"] = _first_valid(
        _numeric(result, "Interest"),
        _numeric(result, "Interest Expense").abs(),
    )

    if "Net Income Common" in result.columns:
        result["Net Income Common"] = _first_valid(
            _numeric(result, "Net Income Common"),
            _numeric(result, "Net Income"),
        )
        # Net Income (당기순이익 전체)는 보존 — Butler/FnGuide는 NI Total 사용.
        # Net Income Common (지배기업 귀속)은 별도 컬럼으로 유지.
    capital_expenditure = _numeric(result, "Capital Expenditure")
    # PPE Sub-Components fallback: sum of 기계장치+차량운반구+건물+토지+... when
    # the aggregated 유형자산의취득 is not reported. This avoids double-counting.
    ppe_sub = _numeric(result, "PPE CapEx Sub-Components")
    _effective_ppe = _first_valid(ppe_capex, ppe_sub)
    # Full sum (PPE + Intangible + InvestmentProperty + ROU) as top priority
    # Butler/FnGuide use comprehensive CAPEX; PPE-only underestimates by 5-10%+
    # Use fillna(0) so that NaN sub-metrics don't poison the sum — e.g. if
    # Investment Property CapEx is present but Intangible CapEx is NaN, we still
    # want PPE + InvProp rather than falling back to PPE-only.
    _has_any_capex = _effective_ppe.notna() | intangible_capex.notna() | investment_property_capex.notna() | rou_capex.notna()
    _capex_full_sum = (
        _effective_ppe.fillna(0) + intangible_capex.fillna(0)
        + investment_property_capex.fillna(0) + rou_capex.fillna(0)
    ).abs().where(_has_any_capex)
    _has_ppe_or_intangible = _effective_ppe.notna() | intangible_capex.notna()
    _capex_ppe_intangible = (_effective_ppe.fillna(0) + intangible_capex.fillna(0)).abs().where(_has_ppe_or_intangible)
    result["Capital Expenditure"] = _first_valid(
        _capex_full_sum,
        _capex_ppe_intangible,
        _effective_ppe.abs(),
        capital_expenditure.abs(),
    )
    # Store PPE-only CapEx separately for Butler/FnGuide comparison
    result["PPE CapEx"] = ppe_capex.abs()

    # Revenue non-negativity guard: YTD→quarterly 변환 오류로 음수가 될 수 있으므로 null 처리
    revenue_final = _numeric(result, "Revenue")
    result["Revenue"] = revenue_final.where(revenue_final.isna() | (revenue_final >= 0), pd.NA)

    # 단위 버그 가드: 티커별 중앙값 대비 1,000배 이상 이상값 → null 처리
    # (DART raw 데이터에서 일부 분기 값이 잘못된 단위로 저장되는 경우 방어)
    _outlier_guard_cols = [
        "Revenue",
        "Operating Income",
        "Net Income",
        "Operating Cash Flow",
        "Capital Expenditure",
        "Cash",
        "Total Assets",
        "Shareholders Equity",
        "Current Assets",
        "Current Liabilities",
    ]
    if "ticker" in result.columns:
        for _col in _outlier_guard_cols:
            if _col not in result.columns:
                continue
            _series = _numeric(result, _col)
            _median_abs = _series.abs().groupby(result["ticker"]).transform("median")
            _obs_count = _series.groupby(result["ticker"]).transform("count")
            _is_outlier = (
                _series.notna()
                & _median_abs.notna()
                & (_median_abs > 0)
                & (_obs_count >= 4)
                & (_series.abs() > _median_abs * 1000)
            )
            result[_col] = _series.where(~_is_outlier, pd.NA)

    return result


_ROW_SCALE_CHECK_COLUMNS = [
    "Revenue",
    "COGS",
    "SG&A",
    "Gross Profit",
    "Operating Income",
    "Net Income",
    "Total Assets",
    "Total Liabilities",
    "Shareholders Equity",
    "Current Assets",
    "Current Liabilities",
    "Cash",
    "AR",
    "AP",
    "Inventory",
    "Operating Cash Flow",
    "Capital Expenditure",
]

_ROW_SCALE_EXCLUDE_COLUMNS = {
    "fiscal_year",
    "fiscal_quarter",
    "Price",
    "Price_M1",
    "Price_M2",
    "Price_M3",
    "Shares",
    "Diluted Shares",
    "Basic Shares",
    "EPS",
    "Diluted EPS",
    "Basic EPS",
}


def _apply_row_scale_normalization(out: pd.DataFrame) -> pd.DataFrame:
    if out.empty or "ticker" not in out.columns or "PeriodEnd" not in out.columns:
        return out

    result = out.copy()
    numeric_cols = []
    for col in result.columns:
        if col in _ROW_SCALE_EXCLUDE_COLUMNS or col.startswith("__"):
            continue
        if pd.api.types.is_numeric_dtype(result[col]):
            numeric_cols.append(col)

    if not numeric_cols:
        return result

    candidate_factors = (1e3, 1e6)
    for ticker, idxs in result.groupby("ticker", sort=False).groups.items():
        ordered = result.loc[idxs].sort_values("PeriodEnd").index.tolist()
        if len(ordered) < 3:
            continue
        for pos in range(1, len(ordered) - 1):
            row_idx = ordered[pos]
            prev_idx = ordered[pos - 1]
            next_idx = ordered[pos + 1]
            supports: dict[float, int] = {factor: 0 for factor in candidate_factors}
            considered = 0
            for col in _ROW_SCALE_CHECK_COLUMNS:
                if col not in result.columns:
                    continue
                prev = pd.to_numeric(result.at[prev_idx, col], errors="coerce")
                cur = pd.to_numeric(result.at[row_idx, col], errors="coerce")
                nxt = pd.to_numeric(result.at[next_idx, col], errors="coerce")
                if pd.isna(prev) or pd.isna(cur) or pd.isna(nxt):
                    continue
                neigh_scale = max((abs(prev) + abs(nxt)) / 2.0, 1.0)
                neighbor_gap = abs(prev - nxt) / neigh_scale
                if neighbor_gap > 0.5:
                    continue
                raw_ratio = abs(cur) / neigh_scale
                considered += 1
                for factor in candidate_factors:
                    adjusted_ratio = abs(cur / factor) / neigh_scale
                    if raw_ratio >= factor * 0.4 and 0.25 <= adjusted_ratio <= 4.0:
                        supports[factor] += 1
            if considered < 3:
                continue
            factor = max(candidate_factors, key=lambda item: supports[item])
            support = supports[factor]
            if support < 3 or support < max(3, int(considered * 0.4)):
                continue
            for col in numeric_cols:
                value = pd.to_numeric(result.at[row_idx, col], errors="coerce")
                if pd.isna(value):
                    continue
                result.at[row_idx, col] = float(value) / factor
    return result


def materialize_financials_quarterly(
    raw_financials: pd.DataFrame,
    filings: pd.DataFrame | None = None,
    ticker_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if raw_financials is None or raw_financials.empty:
        return pd.DataFrame()

    raw = raw_financials.copy()
    for column in (
        "thstrm_amount",
        "thstrm_add_amount",
        "frmtrm_amount",
        "frmtrm_add_amount",
        "frmtrm_q_amount",
        "bfefrmtrm_amount",
        "ord",
    ):
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    metric_info = raw.apply(
        lambda row: _canonical_metric(row.get("account_id"), row.get("account_nm")),
        axis=1,
        result_type="expand",
    )
    raw["canonical_metric"] = metric_info[0]
    raw["metric_priority"] = metric_info[1]
    raw = raw.dropna(subset=["ticker", "bsns_year", "reprt_code"])
    raw["period_end"] = pd.to_datetime(
        raw.get("period_end", pd.Series([pd.NaT] * len(raw), index=raw.index)),
        errors="coerce",
    )
    raw = _select_metric_rows(raw)

    filing_map = pd.DataFrame()
    if filings is not None and not filings.empty:
        filing_map = filings.copy()
        filing_map["filing_date"] = pd.to_datetime(filing_map.get("filing_date"), errors="coerce")
        filing_map["available_date"] = pd.to_datetime(filing_map.get("available_date"), errors="coerce")
        filing_map["period_end"] = pd.to_datetime(filing_map.get("period_end"), errors="coerce")
        filing_map["report_code"] = filing_map.get("report_code", pd.Series(dtype=object)).astype(str)

    name_map = pd.DataFrame()
    if ticker_master is not None and not ticker_master.empty:
        name_map = ticker_master[["ticker", "ticker_name", "industry_name", "sector_name"]].drop_duplicates(subset=["ticker"])

    rows: list[dict[str, object]] = []
    group_cols = ["ticker", "bsns_year", "reprt_code"]
    for (ticker, bsns_year, reprt_code), chunk in raw.groupby(group_cols, dropna=False):
        meta = _REPORT_META.get(str(reprt_code), {})
        raw_period_end = _first_valid_timestamp(chunk.get("period_end"))
        receipt_no = next(
            (
                str(value).strip()
                for value in chunk.get("receipt_no", pd.Series(dtype=object)).tolist()
                if str(value or "").strip()
            ),
            None,
        )
        matched_filing = _matching_filing(
            filing_map,
            ticker=str(ticker),
            reprt_code=str(reprt_code),
            period_end=raw_period_end,
            receipt_no=receipt_no,
        )
        filing_period_end = matched_filing.get("period_end") if matched_filing is not None else pd.NaT
        period_end = _resolve_period_end(
            raw_period_end=raw_period_end,
            filing_period_end=filing_period_end,
            bsns_year=int(bsns_year),
            reprt_code=str(reprt_code),
        )
        row: dict[str, object] = {
            "ticker": str(ticker),
            "market": "kr",
            "term": meta.get("term"),
            "fiscal_year": int(bsns_year),
            "fiscal_quarter": int(meta.get("order", 0)) if meta.get("order") is not None else pd.NA,
            "fiscal_label": f"{int(bsns_year)}Q{int(meta.get('order', 0))}" if meta.get("order") is not None else pd.NA,
            "StatementDate": period_end,
            "PeriodEnd": period_end,
            "PeriodStart": _period_start_from_period_end(period_end),
            "FormType": meta.get("form_type"),
            "FilingDate": pd.NaT,
            "AcceptedAt": pd.NaT,
            "AvailableDate": pd.NaT,
            "AvailabilityMethod": "filed",
            "Source": "dart",
            "collected_at": now_utc_iso(),
            "__bsns_year": int(bsns_year),
            "__report_order": int(meta.get("order", 99)),
        }
        if matched_filing is not None:
            row["FilingDate"] = matched_filing.get("filing_date")
            available_date = matched_filing.get("available_date")
            row["AvailableDate"] = available_date if pd.notna(available_date) else matched_filing.get("filing_date")

        for _, metric_row in chunk.iterrows():
            metric = str(metric_row.get("canonical_metric") or "").strip()
            if not metric:
                continue

            token = _metric_token(metric)
            row[f"__mode_{token}"] = metric_row.get("value_mode")
            row[f"__quarter_{token}"] = metric_row.get("current_amount")
            row[f"__cumulative_{token}"] = metric_row.get("cumulative_amount")

            if metric not in _FLOW_METRICS:
                row[metric] = metric_row.get("current_amount")

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = _finalize_flow_metrics(out)
    if not name_map.empty:
        out = out.merge(name_map, on="ticker", how="left")
        out = out.rename(columns={"ticker_name": "name", "sector_name": "sector", "industry_name": "industry"})
    else:
        out["name"] = pd.NA
        out["sector"] = pd.NA
        out["industry"] = pd.NA

    out = _apply_derived_metric_fallbacks(out)
    out = _apply_row_scale_normalization(out)

    if "AvailableDate" in out.columns:
        out["AvailableDate"] = pd.to_datetime(out["AvailableDate"], errors="coerce").fillna(pd.to_datetime(out["FilingDate"], errors="coerce"))

    if "Net Income Common" in out.columns:
        out["net_income_common"] = pd.to_numeric(out["Net Income Common"], errors="coerce").where(
            pd.to_numeric(out["Net Income Common"], errors="coerce").notna(),
            pd.to_numeric(out.get("Net Income"), errors="coerce"),
        )
    if "Shares" in out.columns:
        shares = pd.to_numeric(out["Shares"], errors="coerce")
        out["Basic Shares"] = shares.where(shares.notna(), pd.to_numeric(out.get("Basic Shares"), errors="coerce"))
        out["basic_shares"] = shares.where(shares.notna(), pd.to_numeric(out.get("basic_shares"), errors="coerce"))

    return (
        out.drop(columns=["__bsns_year", "__report_order", *_REMOVED_OUTPUT_METRICS], errors="ignore")
        .sort_values(["ticker", "PeriodEnd"])
        .reset_index(drop=True)
    )

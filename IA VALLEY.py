import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    FINMA_AVAILABLE = True
except ImportError:
    FINMA_AVAILABLE = False
    print("⚠️ Bibliotecas do FinMA-7B não encontradas. Usando análise estatística padrão.")

class EnhancedFinancialAnalyzer:
    def __init__(self, connection_string):
        """
        Inicializa o analisador financeiro aprimorado
        """
        self.connection_string = connection_string
        self.conn = None
        self.finma_model = None
        self.finma_tokenizer = None
        
        # Mapeamento para classificação de contas
        self.conta_mapping = {
            'ativo_circulante': ['111'],
            'ativo_nao_circulante': ['112'],
            'passivo_circulante': ['211'],
            'passivo_nao_circulante': ['221'],
            'patrimonio_liquido': ['231'],
            'receita_bruta': ['311', '312', '313', '315'],
            'custos_vendas': ['411', '412', '413', '415'],
            'despesas_operacionais': ['511', '512'],
        }
        
        # Configuração do FinMA-7B
        if FINMA_AVAILABLE:
            try:
                print("🔄 Carregando modelo FinMA-7B...")
                self.finma_tokenizer = AutoTokenizer.from_pretrained("ChanceFocus/finma-7b-nlp")
                self.finma_model = AutoModelForCausalLM.from_pretrained(
                    "ChanceFocus/finma-7b-nlp", 
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                print("✅ FinMA-7B carregado com sucesso!")
            except Exception as e:
                print(f"⚠️ Erro ao carregar FinMA-7B: {e}")
                self.finma_model = None
    
    def connect_database(self):
        """Conecta ao banco de dados SQL Server"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            print("✅ Conexão com banco de dados estabelecida!")
            return True
        except Exception as e:
            print(f"❌ Erro na conexão: {e}")
            return False
    
    def get_company_data(self, cod_empresa):
        """
        Extrai dados financeiros da empresa específica utilizando os valores já calculados
        """
        query = """
        SELECT 
            b.[Mês],
            b.Ano,
            b.[Cod Grupo Empresa],
            b.[Cod Empresa],
            b.[Cod Filial],
            b.[Cod Centro Custo],
            b.[Cod Conta Contabil PN],
            p.[DESCRIÇÃO CONTA],
            p.DescN0,
            p.DescN1,
            p.DescN2,
            p.DescN3,
            b.[Conta Cliente],
            b.[Dsc Conta Cliente],
            b.Saldo,
            b.R12,
            b.R1,
            b.KeyEmpresa,
            b.dIndicador,
            i.Indicador as DescIndicador,
            b.R1Indicador,
            b.R12Indicador,
            b.dContaDFC,
            dfc.LinhaDFC,
            dfc.Agrupador,
            dfc.OrdemAgrupador,
            -- Campos para classificação
            CASE 
                WHEN p.DescN0 = 'ATIVO' AND p.DescN1 LIKE '%Circulante%' THEN 'ATIVO_CIRCULANTE'
                WHEN p.DescN0 = 'ATIVO' AND p.DescN1 NOT LIKE '%Circulante%' THEN 'ATIVO_NAO_CIRCULANTE'
                WHEN p.DescN0 = 'PASSIVO' AND p.DescN1 LIKE '%Circulante%' THEN 'PASSIVO_CIRCULANTE'
                WHEN p.DescN0 = 'PASSIVO' AND p.DescN1 NOT LIKE '%Circulante%' THEN 'PASSIVO_NAO_CIRCULANTE'
                WHEN p.DescN1 LIKE '%Patrimônio%' OR p.DescN1 LIKE '%Líquido%' THEN 'PATRIMONIO_LIQUIDO'
                WHEN p.DescN0 LIKE '%RECEITA%' OR p.DescN1 LIKE '%Receita%' THEN 'RECEITAS'
                WHEN p.DescN0 LIKE '%DESPESA%' OR p.DescN1 LIKE '%Despesa%' THEN 'DESPESAS'
                WHEN p.DescN0 LIKE '%CUSTO%' OR p.DescN1 LIKE '%Custo%' THEN 'CUSTOS'
                ELSE 'OUTROS'
            END as GrupoContabil
        FROM dbo.fBalancete_Consolidado b
        LEFT JOIN dbo.dPlanoContasSFV p ON b.[Cod Conta Contabil PN] = p.[CÓD CONTA]
        LEFT JOIN dbo.dIndicador i ON b.dIndicador = i.ID
        LEFT JOIN dbo.dContaDFC dfc ON b.dContaDFC = dfc.CodDFC
        WHERE b.[Cod Empresa] = ?
        AND b.Saldo IS NOT NULL
        ORDER BY b.Ano DESC, b.[Mês] DESC, b.[Cod Conta Contabil PN]
        """
        
        df = pd.read_sql(query, self.conn, params=[cod_empresa])
        
        # Melhora a classificação das contas
        df = self.enhance_account_classification(df)
        
        return df
    
    def enhance_account_classification(self, df):
        """
        Melhora a classificação das contas contábeis baseada na descrição
        """
        df = df.copy()
        
        def improve_classification(row):
            desc = str(row.get('DESCRIÇÃO CONTA', '')).upper()
            grupo_atual = row.get('GrupoContabil', 'OUTROS')

            if any(word in desc for word in ['CAIXA', 'BANCO', 'CONTA CORRENTE', 'APLICAÇÃO']):
                return 'ATIVO_CIRCULANTE'
            elif any(word in desc for word in ['ESTOQUE', 'MERCADORIA', 'MATÉRIA PRIMA']):
                return 'ATIVO_CIRCULANTE'
            elif any(word in desc for word in ['CLIENTE', 'DUPLICATA RECEBER', 'CONTAS RECEBER']):
                return 'ATIVO_CIRCULANTE'
            elif any(word in desc for word in ['IMÓVEL', 'VEÍCULO', 'MÁQUINA', 'EQUIPAMENTO', 'IMOBILIZADO']):
                return 'ATIVO_NAO_CIRCULANTE'
            elif any(word in desc for word in ['FORNECEDOR', 'DUPLICATA PAGAR', 'CONTAS PAGAR']):
                return 'PASSIVO_CIRCULANTE'
            elif any(word in desc for word in ['EMPRÉSTIMO', 'FINANCIAMENTO']) and 'LONGO PRAZO' not in desc:
                return 'PASSIVO_CIRCULANTE'
            elif any(word in desc for word in ['CAPITAL', 'RESERVA', 'LUCRO', 'PREJUÍZO']):
                return 'PATRIMONIO_LIQUIDO'
            elif any(word in desc for word in ['RECEITA', 'VENDA', 'FATURAMENTO']):
                return 'RECEITAS'
            elif any(word in desc for word in ['CUSTO', 'CMV', 'CPV']):
                return 'CUSTOS'
            elif any(word in desc for word in ['DESPESA', 'GASTO']):
                return 'DESPESAS'
            else:
                return grupo_atual
        
        df['GrupoContabil'] = df.apply(improve_classification, axis=1)
        
        return df
    
    def analyze_existing_indicators(self, df):
        """
        Analisa indicadores já calculados presentes na base
        """
        indicators_analysis = []
        
        # Filtra registros com indicadores calculados
        df_indicators = df[df['dIndicador'].notna() & df['R1Indicador'].notna()].copy()
        
        if not df_indicators.empty:
            # Agrupa por indicador e período
            for (indicador, periodo), group in df_indicators.groupby(['dIndicador', 'Ano', 'Mês']):
                # Pega o primeiro registro do grupo (todos devem ter os mesmos valores de indicador)
                row = group.iloc[0]
                
                analysis = {
                    'periodo': f"{row['Ano']}-{int(row['Mês']):02d}",
                    'ano': row['Ano'],
                    'mes': row['Mês'],
                    'codigo_indicador': row['dIndicador'],
                    'descricao_indicador': row.get('DescIndicador', 'N/A'),
                    'r1_indicador': row['R1Indicador'],
                    'r12_indicador': row['R12Indicador'],
                    'tipo': 'MENSAL' if pd.notna(row['R1Indicador']) else 'ANUAL'
                }
                
                indicators_analysis.append(analysis)
        
        return pd.DataFrame(indicators_analysis)
    
    def analyze_dfc_data(self, df):
        """
        Analisa dados da Demonstração do Fluxo de Caixa já calculados
        """
        dfc_analysis = []
        
        # Filtra registros com DFC
        df_dfc = df[df['dContaDFC'].notna()].copy()
        
        if not df_dfc.empty:
            # Ordena por período e ordem do DFC
            df_dfc['periodo'] = df_dfc['Ano'].astype(str) + '-' + df_dfc['Mês'].astype(str).str.zfill(2)
            df_dfc = df_dfc.sort_values(['periodo', 'OrdemAgrupador'])
            
            for periodo in df_dfc['periodo'].unique():
                periodo_data = df_dfc[df_dfc['periodo'] == periodo]
                
                for agrupador in periodo_data['Agrupador'].unique():
                    agrupador_data = periodo_data[periodo_data['Agrupador'] == agrupador]
                    
                    # Soma os valores do mesmo agrupador
                    saldo_agrupador = agrupador_data['Saldo'].sum()
                    
                    dfc_analysis.append({
                        'periodo': periodo,
                        'agrupador_dfc': agrupador,
                        'valor': saldo_agrupador,
                        'qtd_linhas': len(agrupador_data)
                    })
        
        return pd.DataFrame(dfc_analysis)
    
    def calculate_variations_existing_data(self, df):
        """
        Calcula variações baseadas nos dados já existentes (R1, R12)
        """
        variations = []
        
        # Agrupa por conta contábil
        for conta in df['Cod Conta Contabil PN'].unique():
            conta_data = df[df['Cod Conta Contabil PN'] == conta].copy()
            
            # Ordena por período
            conta_data = conta_data.sort_values(['Ano', 'Mês'])
            
            if len(conta_data) > 1:
                # Calcula variações usando R1 (último mês) se disponível
                if 'R1' in conta_data.columns and not conta_data['R1'].isna().all():
                    # Usa R1 para variação mensal
                    for i in range(1, len(conta_data)):
                        current = conta_data.iloc[i]
                        previous = conta_data.iloc[i-1]
                        
                        if pd.notna(current['R1']) and pd.notna(previous['R1']):
                            variacao_pct = ((current['R1'] - previous['R1']) / abs(previous['R1'])) * 100 if previous['R1'] != 0 else 0
                            variacao_abs = current['R1'] - previous['R1']
                            
                            variations.append({
                                'conta': current['Cod Conta Contabil PN'],
                                'descricao': current.get('DESCRIÇÃO CONTA', 'N/A'),
                                'grupo_contabil': current.get('GrupoContabil', 'N/A'),
                                'periodo': f"{current['Ano']}-{int(current['Mês']):02d}",
                                'saldo_atual': current['R1'],
                                'saldo_anterior': previous['R1'],
                                'variacao_pct': variacao_pct,
                                'variacao_abs': variacao_abs,
                                'tipo': 'MENSAL_R1'
                            })
        
        return pd.DataFrame(variations)
    
    def generate_financial_summary_existing_data(self, df):
        """
        Gera resumo financeiro utilizando dados já consolidados
        """
        summary = {}
        
        # Utiliza os grupos já classificados para somar os saldos
        grupos = df['GrupoContabil'].unique()
        
        for grupo in grupos:
            grupo_data = df[df['GrupoContabil'] == grupo]
            summary[grupo.lower()] = grupo_data['Saldo'].sum()
        
        # Adiciona informações básicas
        summary['periodo_inicio'] = f"{df['Ano'].min()}-{df['Mês'].min():02d}"
        summary['periodo_fim'] = f"{df['Ano'].max()}-{df['Mês'].max():02d}"
        summary['total_contas'] = df['Cod Conta Contabil PN'].nunique()
        
        return summary
    
    def analyze_cash_flow_existing(self, df):
        """
        Analisa fluxo de caixa baseado em contas de caixa já identificadas
        """
        caixa_data = df[df['DESCRIÇÃO CONTA'].str.contains('CAIXA|BANCO', case=False, na=False)]
        
        if caixa_data.empty:
            return {
                'saldo_atual': 0,
                'movimentacao_total': 0,
                'periodos_analisados': 0
            }
        
        # Ordena por período
        caixa_data = caixa_data.sort_values(['Ano', 'Mês'])
        
        return {
            'saldo_atual': caixa_data['Saldo'].iloc[-1] if len(caixa_data) > 0 else 0,
            'saldo_inicial': caixa_data['Saldo'].iloc[0] if len(caixa_data) > 0 else 0,
            'variacao_total': caixa_data['Saldo'].iloc[-1] - caixa_data['Saldo'].iloc[0] if len(caixa_data) > 1 else 0,
            'periodos_analisados': len(caixa_data),
            'media_mensal': caixa_data['Saldo'].mean()
        }
    
    def generate_analysis_report(self, cod_empresa, df, indicators_df, variations_df, dfc_df, financial_summary, cash_flow_analysis):
        """
        Gera relatório de análise completo baseado nos dados existentes
        """
        report_lines = []
        
        report_lines.append("="*100)
        report_lines.append("🌾 RELATÓRIO DE ANÁLISE FINANCEIRA - VALLEY IRRIGAÇÃO")
        report_lines.append("📊 Baseado em dados consolidados existentes")
        report_lines.append("="*100)
        
        # Informações básicas
        report_lines.append(f"\n📋 INFORMAÇÕES GERAIS:")
        report_lines.append(f"   • Empresa: {cod_empresa}")
        report_lines.append(f"   • Período analisado: {financial_summary['periodo_inicio']} a {financial_summary['periodo_fim']}")
        report_lines.append(f"   • Total de contas analisadas: {financial_summary['total_contas']}")
        
        # Resumo por grupos contábeis
        report_lines.append(f"\n💰 RESUMO PATRIMONIAL:")
        for grupo, valor in financial_summary.items():
            if any(key in grupo for key in ['ativo', 'passivo', 'patrimonio']):
                report_lines.append(f"   • {grupo.upper().replace('_', ' ')}: R$ {valor:>15,.2f}")
        
        # Análise de indicadores
        if not indicators_df.empty:
            report_lines.append(f"\n📈 INDICADORES FINANCEIROS:")
            for indicador in indicators_df['codigo_indicador'].unique():
                ind_data = indicators_df[indicators_df['codigo_indicador'] == indicador]
                descricao = ind_data['descricao_indicador'].iloc[0]
                
                report_lines.append(f"\n   {descricao}:")
                for _, row in ind_data.iterrows():
                    report_lines.append(f"      • {row['periodo']}: {row['r1_indicador']:.2f}")
        
        # Análise DFC
        if not dfc_df.empty:
            report_lines.append(f"\n💸 DEMONSTRAÇÃO DO FLUXO DE CAIXA:")
            for periodo in dfc_df['periodo'].unique():
                periodo_data = dfc_df[dfc_df['periodo'] == periodo]
                report_lines.append(f"\n   Período {periodo}:")
                for _, row in periodo_data.iterrows():
                    report_lines.append(f"      • {row['agrupador_dfc']}: R$ {row['valor']:>15,.2f}")
        
        # Análise de variações
        if not variations_df.empty:
            report_lines.append(f"\n📊 VARIAÇÕES SIGNIFICATIVAS:")
            variacoes_significativas = variations_df[abs(variations_df['variacao_pct']) > 30]
            
            for _, var in variacoes_significativas.iterrows():
                sinal = "📈" if var['variacao_pct'] > 0 else "📉"
                report_lines.append(f"   {sinal} {var['descricao'][:40]}...: {var['variacao_pct']:+.1f}%")
        
        # Situação do caixa
        report_lines.append(f"\n💵 SITUAÇÃO DO CAIXA:")
        report_lines.append(f"   • Saldo atual: R$ {cash_flow_analysis['saldo_atual']:>15,.2f}")
        report_lines.append(f"   • Variação no período: R$ {cash_flow_analysis['variacao_total']:>11,.2f}")
        report_lines.append(f"   • Períodos analisados: {cash_flow_analysis['periodos_analisados']:>10}")
        
        report_lines.append("\n" + "="*100)
        
        return "\n".join(report_lines)
    
    def run_analysis(self):
        """
        Executa análise completa baseada em dados existentes
        """
        print("\n" + "="*80)
        print("🌾 ANÁLISE FINANCEIRA - VALLEY IRRIGAÇÃO")
        print("📊 Utilizando dados consolidados existentes")
        print("="*80)
        
        if not self.connect_database():
            return
        
        try:
            cod_empresa = input("\n📊 Digite o código da empresa para análise: ").strip()
            if not cod_empresa:
                print("❌ Código da empresa é obrigatório!")
                return
            
            print(f"\n🔍 Extraindo dados da empresa {cod_empresa}...")
            df = self.get_company_data(cod_empresa)
            
            if df.empty:
                print(f"❌ Nenhum dado encontrado para a empresa {cod_empresa}")
                return
            
            print("📈 Analisando indicadores existentes...")
            indicators_df = self.analyze_existing_indicators(df)
            
            print("💸 Analisando Demonstração do Fluxo de Caixa...")
            dfc_df = self.analyze_dfc_data(df)
            
            print("📊 Calculando variações...")
            variations_df = self.calculate_variations_existing_data(df)
            
            print("💰 Gerando resumo financeiro...")
            financial_summary = self.generate_financial_summary_existing_data(df)
            
            print("💵 Analisando situação do caixa...")
            cash_flow_analysis = self.analyze_cash_flow_existing(df)
            
            print("📋 Gerando relatório final...")
            report = self.generate_analysis_report(
                cod_empresa, df, indicators_df, variations_df, 
                dfc_df, financial_summary, cash_flow_analysis
            )
            
            # Exibe o relatório
            print("\n" + report)
            
            # Opção para salvar
            save = input("\n💾 Deseja salvar o relatório? (s/n): ").lower()
            if save == 's':
                filename = f"relatorio_empresa_{cod_empresa}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"✅ Relatório salvo como: {filename}")
            
        except Exception as e:
            print(f"❌ Erro durante a análise: {e}")
        finally:
            if self.conn:
                self.conn.close()

def main():
    """
    Função principal
    """
    print("🚀 Iniciando Sistema de Análise Financeira...")
    
    # Configuração de conexão
    server = input("🖥️  Servidor (Enter para localhost): ").strip() or "localhost"
    database = input("🗄️  Banco de dados: ").strip()
    
    if not database:
        print("❌ Nome do banco de dados é obrigatório!")
        return
    
    connection_string = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={server};
    DATABASE={database};
    Trusted_Connection=yes;
    """
    
    try:
        analyzer = EnhancedFinancialAnalyzer(connection_string)
        analyzer.run_analysis()
    except Exception as e:
        print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()
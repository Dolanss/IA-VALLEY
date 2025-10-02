import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    FINMA_AVAILABLE = True
except ImportError:
    FINMA_AVAILABLE = False
    print("⚠️ Bibliotecas do FinMA-7B não encontradas. Usando análise estatística padrão.")

class FinancialAnalyzerWithFinMA:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.conn = None
        self.finma_model = None
        self.finma_tokenizer = None
        
        # Tabelas de referência
        self.dfc_mapping = self._create_dfc_mapping()
        self.indicators_mapping = self._create_indicators_mapping()
        
        if FINMA_AVAILABLE:
            self._load_finma_model()
    
    def _create_dfc_mapping(self):
        """Cria mapeamento de DFC"""
        return {
            'D.1': {'linha': '(+) Resultado do Exercício', 'agrupador': 'Resultado Líquido Ajustado', 'ordem': 1},
            'D.2': {'linha': '(-) Depreciação', 'agrupador': 'Resultado Líquido Ajustado', 'ordem': 1},
            'D.3': {'linha': '(=) Geração Bruta de Caixa', 'agrupador': 'Resultado Líquido Ajustado', 'ordem': 1},
            'D.4': {'linha': 'Variação de NCG', 'agrupador': 'Variação da NCG', 'ordem': 2},
            'D.5': {'linha': '(+/-) Var Fornecedores', 'agrupador': 'Variação da NCG', 'ordem': 2},
            'D.6': {'linha': '(+/-) Var Outros Passivos Operacionais', 'agrupador': 'Variação da NCG', 'ordem': 2},
            'D.7': {'linha': '(+/-) Var Contas a Receber', 'agrupador': 'Variação da NCG', 'ordem': 2},
            'D.8': {'linha': '(+/-) Var Estoques', 'agrupador': 'Variação da NCG', 'ordem': 2},
            'D.9': {'linha': '(+/-) Var Outros Ativos Operacionais', 'agrupador': 'Variação da NCG', 'ordem': 2},
            'D.10': {'linha': '(=) Fluxo de Caixa da Operação', 'agrupador': '(=) Fluxo de Caixa da Operação', 'ordem': 3},
            'D.11': {'linha': '(+/-) CAPEX', 'agrupador': '( +/- ) Atividades de Investimento', 'ordem': 4},
            'D.12': {'linha': '(+/-) Intangível', 'agrupador': '( +/- ) Atividades de Investimento', 'ordem': 4},
            'D.13': {'linha': '(+/-) Outros Ativos/Passivos Não Circulantes', 'agrupador': '( +/- ) Atividades de Investimento', 'ordem': 4},
            'D.14': {'linha': '(=) Fluxo de Caixa Pós-Investimentos', 'agrupador': '(=) Fluxo de Caixa Pós-Investimentos', 'ordem': 5},
            'D.15': {'linha': '(+/-) Empréstimos e Financiamento CP', 'agrupador': '( +/- ) Atividades de Financiamento', 'ordem': 6},
            'D.16': {'linha': '(+/-) Empréstimos e Financiamento LP', 'agrupador': '( +/- ) Atividades de Financiamento', 'ordem': 6},
            'D.17': {'linha': '(+/-) Recebíveis e Investimentos LP', 'agrupador': '( +/- ) Atividades de Financiamento', 'ordem': 6},
            'D.18': {'linha': '(+/-)  Patrimonial, Ingralização e Distr Capital', 'agrupador': '( +/- ) Atividades de Financiamento', 'ordem': 6},
            'D.19': {'linha': '(=) Fluxo de Caixa', 'agrupador': '(=) Fluxo de Caixa', 'ordem': 7},
            'D.20': {'linha': 'Saldo Inicial de Caixa', 'agrupador': 'Saldo Inicial de Caixa', 'ordem': 8},
            'D.21': {'linha': 'Saldo Final de Caixa', 'agrupador': 'Saldo Final de Caixa', 'ordem': 9}
        }
    
    def _create_indicators_mapping(self):
        """Cria mapeamento completo de indicadores"""
        return {
            1: {'nome': '% Despesas fixas / ROL', 'tipo': '% ROL', 'melhor': 'Menor', 'unidade': '%'},
            2: {'nome': 'Margem de Contribuição', 'tipo': 'Rentabilidade', 'melhor': 'Maior', 'unidade': '%'},
            3: {'nome': 'Liquidez Corrente', 'tipo': 'Liquidez', 'melhor': 'Maior', 'unidade': '#'},
            4: {'nome': 'Liquidez Seca', 'tipo': 'Liquidez', 'melhor': 'Maior', 'unidade': '#'},
            5: {'nome': 'Liquidez Geral', 'tipo': 'Liquidez', 'melhor': 'Maior', 'unidade': '#'},
            6: {'nome': 'Grau de Imobilização s/ Ativo', 'tipo': 'Estrutura', 'melhor': 'Menor', 'unidade': '%'},
            7: {'nome': 'Participação Capital Terceiros (PCT)', 'tipo': 'Estrutura', 'melhor': 'Menor', 'unidade': '%'},
            8: {'nome': 'Participação Capital Próprio (Equity)', 'tipo': 'Estrutura', 'melhor': 'Maior', 'unidade': '%'},
            9: {'nome': 'Imobilização Patrimônio Líquido (IPL)', 'tipo': 'Estrutura', 'melhor': 'Menor', 'unidade': '%'},
            10: {'nome': 'Retorno Sobre Ativos (ROA)', 'tipo': 'Rentabilidade', 'melhor': 'Maior', 'unidade': '%'},
            11: {'nome': 'Retorno Sobre PL (ROE)', 'tipo': 'Rentabilidade', 'melhor': 'Maior', 'unidade': '%'},
            12: {'nome': 'Margem Líquida', 'tipo': 'Rentabilidade', 'melhor': 'Maior', 'unidade': '%'},
            13: {'nome': 'Giro de Ativos', 'tipo': 'Estrutura', 'melhor': 'Maior', 'unidade': '#'},
            14: {'nome': '% Despesas sobre Vendas (c/Desp Fin.)', 'tipo': '% ROL', 'melhor': 'Menor', 'unidade': '%'},
            28: {'nome': 'Giro de Estoque de Peças', 'tipo': 'Peças', 'melhor': 'Maior', 'unidade': '#'},
            29: {'nome': 'EBITDA', 'tipo': 'Rentabilidade', 'melhor': 'Maior', 'unidade': '%'},
            30: {'nome': 'Margem Bruta', 'tipo': 'Rentabilidade', 'melhor': 'Maior', 'unidade': '%'},
            31: {'nome': '% Despesas sobre Vendas (s/Desp Fin.)', 'tipo': '% ROL', 'melhor': 'Menor', 'unidade': '%'}
        }
    
    def _load_finma_model(self):
        """Carrega modelo FinMA-7B"""
        try:
            print("🔄 Carregando modelo FinMA-7B...")
            self.finma_tokenizer = AutoTokenizer.from_pretrained("ChanceFocus/finma-7b-nlp")
            self.finma_model = AutoModelForCausalLM.from_pretrained(
                "ChanceFocus/finma-7b-nlp",
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            print("✅ FinMA-7B carregado com sucesso!")
        except Exception as e:
            print(f"⚠️ Erro ao carregar FinMA-7B: {e}")
            self.finma_model = None
    
    def connect_database(self):
        """Conecta ao banco de dados"""
        try:
            self.conn = pyodbc.connect(self.connection_string)
            print("✅ Conexão estabelecida!")
            return True
        except Exception as e:
            print(f"❌ Erro na conexão: {e}")
            return False
    
    def get_company_data(self, cod_empresa):
        """Extrai dados da empresa"""
        query = """
        SELECT 
            b.[Mês], b.Ano, b.[Cod Empresa], b.[Cod Conta Contabil PN],
            p.[DESCRIÇÃO CONTA], p.DescN0, p.DescN1, p.DescN2,
            b.Saldo, b.R12, b.R1, b.R1_12,
            b.dIndicador, b.R1Indicador, b.R12Indicador,
            b.dContaDFC
        FROM dbo.fBalancete_Consolidado b
        LEFT JOIN dbo.dPlanoContasSFV p ON b.[Cod Conta Contabil PN] = p.[CÓD CONTA]
        WHERE b.[Cod Empresa] = ?
        ORDER BY b.Ano DESC, b.[Mês] DESC
        """
        df = pd.read_sql(query, self.conn, params=[cod_empresa])
        
        # Enriquece dados com mapeamentos internos
        df = self._enrich_dfc_from_mapping(df)
        df = self._enrich_indicators_from_mapping(df)
        
        return df
    
    def _enrich_dfc_from_mapping(self, df):
        """Enriquece dados com mapeamento DFC interno"""
        def get_dfc_info(cod_dfc):
            if pd.isna(cod_dfc):
                return pd.Series([None, None, None])
            cod_str = str(cod_dfc).strip()
            if cod_str in self.dfc_mapping:
                info = self.dfc_mapping[cod_str]
                return pd.Series([info['linha'], info['agrupador'], info['ordem']])
            return pd.Series([None, None, None])
        
        df[['LinhaDFC', 'Agrupador', 'OrdemAgrupador']] = df['dContaDFC'].apply(get_dfc_info)
        return df
    
    def _enrich_indicators_from_mapping(self, df):
        """Enriquece dados com mapeamento de indicadores interno"""
        def get_indicator_name(cod_ind):
            if pd.isna(cod_ind):
                return None
            try:
                cod_int = int(cod_ind)
                if cod_int in self.indicators_mapping:
                    return self.indicators_mapping[cod_int]['nome']
            except:
                pass
            return None
        
        df['DescIndicador'] = df['dIndicador'].apply(get_indicator_name)
        return df
    
    def extract_value(self, row):
        """Extrai o valor correto: R1 para DFC/Indicador, Saldo para outros"""
        if pd.notna(row.get('dContaDFC')) or pd.notna(row.get('dIndicador')):
            return row.get('R1', 0) if pd.notna(row.get('R1')) else 0
        return row.get('Saldo', 0) if pd.notna(row.get('Saldo')) else 0
    
    def analyze_large_variations(self, df):
        """Analisa variações superiores a 100%"""
        variations = []
        
        df['periodo'] = df['Ano'].astype(str) + '-' + df['Mês'].astype(str).str.zfill(2)
        df['valor_analise'] = df.apply(self.extract_value, axis=1)
        
        for conta in df['Cod Conta Contabil PN'].unique():
            conta_data = df[df['Cod Conta Contabil PN'] == conta].sort_values('periodo')
            
            if len(conta_data) < 2:
                continue
            
            for i in range(1, len(conta_data)):
                atual = conta_data.iloc[i]
                anterior = conta_data.iloc[i-1]
                
                val_atual = atual['valor_analise']
                val_anterior = anterior['valor_analise']
                
                if val_anterior != 0 and abs(val_atual) > 0:
                    var_pct = ((val_atual - val_anterior) / abs(val_anterior)) * 100
                    
                    if abs(var_pct) > 100:
                        variations.append({
                            'conta': conta,
                            'descricao': atual.get('DESCRIÇÃO CONTA', 'N/A'),
                            'periodo': atual['periodo'],
                            'valor_anterior': val_anterior,
                            'valor_atual': val_atual,
                            'variacao_pct': var_pct,
                            'variacao_abs': val_atual - val_anterior
                        })
        
        return pd.DataFrame(variations).sort_values('variacao_pct', key=abs, ascending=False)
    
    def analyze_dfc(self, df):
        """Analisa Demonstração do Fluxo de Caixa"""
        df_dfc = df[df['dContaDFC'].notna()].copy()
        
        if df_dfc.empty:
            return pd.DataFrame()
        
        df_dfc['periodo'] = df_dfc['Ano'].astype(str) + '-' + df_dfc['Mês'].astype(str).str.zfill(2)
        df_dfc['valor_dfc'] = df_dfc['R1'].fillna(0)
        
        dfc_summary = df_dfc.groupby(['periodo', 'Agrupador', 'OrdemAgrupador']).agg({
            'valor_dfc': 'sum'
        }).reset_index().sort_values(['periodo', 'OrdemAgrupador'])
        
        return dfc_summary
    
    def analyze_indicators(self, df):
        """Analisa indicadores financeiros"""
        df_ind = df[df['dIndicador'].notna()].copy()
        
        if df_ind.empty:
            return pd.DataFrame()
        
        df_ind['periodo'] = df_ind['Ano'].astype(str) + '-' + df_ind['Mês'].astype(str).str.zfill(2)
        
        indicators_summary = df_ind.groupby(['periodo', 'dIndicador', 'DescIndicador']).agg({
            'R1Indicador': 'first',
            'R12Indicador': 'first'
        }).reset_index()
        
        return indicators_summary
    
    def generate_finma_insight(self, context_text):
        """Gera insights usando FinMA-7B"""
        if not self.finma_model or not FINMA_AVAILABLE:
            return "Análise via FinMA-7B não disponível."
        
        try:
            prompt = f"""Você é um analista financeiro especializado. Analise os seguintes dados financeiros e forneça insights concisos e práticos:

{context_text}

Forneça uma análise objetiva focada em:
1. Principais riscos identificados
2. Oportunidades de melhoria
3. Recomendações prioritárias

Análise:"""
            
            inputs = self.finma_tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
            inputs = {k: v.to(self.finma_model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.finma_model.generate(
                    **inputs,
                    max_new_tokens=300,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = self.finma_tokenizer.decode(outputs[0], skip_special_tokens=True)
            analysis = response.split("Análise:")[-1].strip()
            
            return analysis if analysis else "Análise inconclusiva."
            
        except Exception as e:
            return f"Erro ao gerar insight: {str(e)}"
    
    def generate_report(self, cod_empresa, variations_df, dfc_df, indicators_df):
        """Gera relatório completo com insights do FinMA"""
        report = []
        report.append("="*100)
        report.append("🌾 RELATÓRIO DE ANÁLISE FINANCEIRA - VALLEY IRRIGAÇÃO")
        report.append(f"📊 Empresa: {cod_empresa} | Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        report.append("="*100)
        
        # 1. VARIAÇÕES SIGNIFICATIVAS (>100%)
        report.append("\n📈 VARIAÇÕES SIGNIFICATIVAS (>100%)")
        report.append("-"*100)
        
        if not variations_df.empty:
            top_variations = variations_df.head(10)
            for _, var in top_variations.iterrows():
                sinal = "⬆️" if var['variacao_pct'] > 0 else "⬇️"
                report.append(f"{sinal} {var['descricao'][:50]}")
                report.append(f"   Período: {var['periodo']} | Variação: {var['variacao_pct']:+.1f}%")
                report.append(f"   Anterior: R$ {var['valor_anterior']:,.2f} → Atual: R$ {var['valor_atual']:,.2f}")
                report.append("")
            
            # Insight FinMA sobre variações
            context = f"Variações superiores a 100%:\n"
            for _, var in top_variations.head(5).iterrows():
                context += f"- {var['descricao']}: {var['variacao_pct']:+.1f}%\n"
            
            report.append("🤖 INSIGHT FinMA - Variações:")
            report.append(self.generate_finma_insight(context))
        else:
            report.append("Nenhuma variação superior a 100% identificada.")
        
        # 2. ANÁLISE DFC
        report.append("\n\n💸 DEMONSTRAÇÃO DO FLUXO DE CAIXA")
        report.append("-"*100)
        
        if not dfc_df.empty:
            ultimo_periodo = dfc_df['periodo'].max()
            dfc_periodo = dfc_df[dfc_df['periodo'] == ultimo_periodo]
            
            report.append(f"Período: {ultimo_periodo}\n")
            for agrupador in dfc_periodo['Agrupador'].unique():
                agrup_data = dfc_periodo[dfc_periodo['Agrupador'] == agrupador]
                total = agrup_data['valor_dfc'].sum()
                report.append(f"  {agrupador}: R$ {total:>15,.2f}")
            
            # Insight FinMA sobre DFC
            context_dfc = f"Fluxo de Caixa ({ultimo_periodo}):\n"
            for agrupador in dfc_periodo['Agrupador'].unique():
                total = dfc_periodo[dfc_periodo['Agrupador'] == agrupador]['valor_dfc'].sum()
                context_dfc += f"- {agrupador}: R$ {total:,.2f}\n"
            
            report.append("\n🤖 INSIGHT FinMA - Fluxo de Caixa:")
            report.append(self.generate_finma_insight(context_dfc))
        else:
            report.append("Dados de DFC não disponíveis.")
        
        # 3. INDICADORES FINANCEIROS
        report.append("\n\n📊 INDICADORES FINANCEIROS PRINCIPAIS")
        report.append("-"*100)
        
        if not indicators_df.empty:
            ultimo_periodo = indicators_df['periodo'].max()
            ind_periodo = indicators_df[indicators_df['periodo'] == ultimo_periodo]
            
            report.append(f"Período: {ultimo_periodo}\n")
            for _, ind in ind_periodo.iterrows():
                cod_ind = int(ind['dIndicador'])
                if cod_ind in self.indicators_mapping:
                    info = self.indicators_mapping[cod_ind]
                    report.append(f"  {info['nome']}: {ind['R1Indicador']:.2f}")
                    report.append(f"    Categoria: {info['tipo']} | Melhor: {info['melhor']}")
                    report.append("")
            
            # Insight FinMA sobre indicadores
            context_ind = f"Indicadores Financeiros ({ultimo_periodo}):\n"
            for _, ind in ind_periodo.head(8).iterrows():
                cod_ind = int(ind['dIndicador'])
                if cod_ind in self.indicators_mapping:
                    nome = self.indicators_mapping[cod_ind]['nome']
                    context_ind += f"- {nome}: {ind['R1Indicador']:.2f}\n"
            
            report.append("🤖 INSIGHT FinMA - Indicadores:")
            report.append(self.generate_finma_insight(context_ind))
        else:
            report.append("Dados de indicadores não disponíveis.")
        
        report.append("\n" + "="*100)
        report.append("Fim do Relatório")
        report.append("="*100)
        
        return "\n".join(report)
    
    def run_analysis(self):
        """Executa análise completa"""
        print("\n🚀 SISTEMA DE ANÁLISE FINANCEIRA COM FinMA-7B")
        print("="*80)
        
        if not self.connect_database():
            return
        
        try:
            cod_empresa = input("\n📊 Código da empresa: ").strip()
            if not cod_empresa:
                print("❌ Código obrigatório!")
                return
            
            print(f"\n🔍 Extraindo dados da empresa {cod_empresa}...")
            df = self.get_company_data(cod_empresa)
            
            if df.empty:
                print(f"❌ Nenhum dado encontrado para empresa {cod_empresa}")
                return
            
            print("📊 Analisando variações superiores a 100%...")
            variations_df = self.analyze_large_variations(df)
            
            print("💸 Analisando Fluxo de Caixa...")
            dfc_df = self.analyze_dfc(df)
            
            print("📈 Analisando indicadores financeiros...")
            indicators_df = self.analyze_indicators(df)
            
            print("📝 Gerando relatório com insights FinMA...")
            report = self.generate_report(cod_empresa, variations_df, dfc_df, indicators_df)
            
            print("\n" + report)
            
            save = input("\n💾 Salvar relatório? (s/n): ").lower()
            if save == 's':
                filename = f"relatorio_finma_{cod_empresa}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"✅ Relatório salvo: {filename}")
                
        except Exception as e:
            print(f"❌ Erro: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.conn:
                self.conn.close()

def main():
    print("🌾 Sistema de Análise Financeira - Valley Irrigação")
    print("="*80)
    
    server = input("🖥️  Servidor (Enter = localhost): ").strip() or "localhost"
    database = input("🗄️  Banco de dados: ").strip()
    
    if not database:
        print("❌ Nome do banco é obrigatório!")
        return
    
    connection_string = f"""
    DRIVER={{ODBC Driver 17 for SQL Server}};
    SERVER={server};
    DATABASE={database};
    Trusted_Connection=yes;
    """
    
    analyzer = FinancialAnalyzerWithFinMA(connection_string)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

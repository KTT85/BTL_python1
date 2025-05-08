import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import re
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

class StatsAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        self.df = None
        self.numeric_columns = None

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('analysis.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        try:
            self.logger.info("Loading data from results.csv...")
            self.df = pd.read_csv('results.csv')
            self.df.replace("N/a", np.nan, inplace=True)
            self.logger.info(f"Loaded {len(self.df)} rows of data")

            def convert_age(age_str):
                try:
                    years, days = map(int, str(age_str).split('-'))
                    return years + (days / 365)
                except:
                    return pd.to_numeric(age_str, errors='coerce')

            self.df['Age'] = self.df['Age'].apply(convert_age).round(2)

            playing_time_cols = ['MP', 'Starts', 'Minutes']
            for col in playing_time_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            exclude_cols = ['Player', 'Nation', 'Team', 'Position']
            self.numeric_columns = self.df.select_dtypes(include=[np.number]).columns
            self.numeric_columns = [col for col in self.numeric_columns if col not in exclude_cols]

            self.logger.info(f"Found {len(self.numeric_columns)} numeric columns for analysis")
            self.logger.info(f"Columns: {', '.join(self.numeric_columns)}")

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def find_top_bottom_players(self):
        with open(self.output_dir / 'top_3.txt', 'w', encoding='utf-8') as f:
            all_stats = ['Age', 'MP', 'Starts', 'Minutes']
            all_stats.extend([col for col in self.numeric_columns if col not in all_stats])

            for stat in all_stats:
                f.write(f"\n{'='*50}\n")
                f.write(f"Statistic: {stat}\n")
                f.write(f"{'-'*50}\n")

                f.write("\nTop 3 Players:\n")
                top_3 = self.df.nlargest(3, stat)
                for _, row in top_3.iterrows():
                    f.write(f"{row['Player']}: {row[stat]:.2f}\n")

                f.write("\nBottom 3 Players:\n")
                bottom_3 = self.df.nsmallest(3, stat)
                for _, row in bottom_3.iterrows():
                    f.write(f"{row['Player']}: {row[stat]:.2f}\n")

    def calculate_statistics(self):
        try:
            team_stats = []
            player_stats = []

            all_stats = ['Age', 'MP', 'Starts', 'Minutes']
            all_stats.extend([col for col in self.numeric_columns if col not in all_stats])

            all_stats_dict = {'Team': 'all'}
            for stat in all_stats:
                all_stats_dict[f'Median of {stat}'] = self.df[stat].median()
                all_stats_dict[f'Mean of {stat}'] = self.df[stat].mean()
                all_stats_dict[f'Std of {stat}'] = self.df[stat].std()
            team_stats.append(all_stats_dict)

            for team in self.df['Team'].dropna().unique():
                team_data = self.df[self.df['Team'] == team]
                stats = {'Team': team}
                for stat in all_stats:
                    stats[f'Median of {stat}'] = team_data[stat].median()
                    stats[f'Mean of {stat}'] = team_data[stat].mean()
                    stats[f'Std of {stat}'] = team_data[stat].std()
                team_stats.append(stats)

            all_player_stats = {'Player': 'all'}
            for stat in all_stats:
                all_player_stats[f'Median of {stat}'] = self.df[stat].median()
                all_player_stats[f'Mean of {stat}'] = self.df[stat].mean()
                all_player_stats[f'Std of {stat}'] = self.df[stat].std()
            player_stats.append(all_player_stats)

            for _, row in self.df.iterrows():
                stats = {'Player': row['Player']}
                for stat in all_stats:
                    stats[f'Median of {stat}'] = row[stat]
                    stats[f'Mean of {stat}'] = row[stat]
                    stats[f'Std of {stat}'] = 0
                player_stats.append(stats)

            team_df = pd.DataFrame(team_stats).set_index('Team')
            player_df = pd.DataFrame(player_stats).set_index('Player')

            column_groups = []
            for stat in all_stats:
                column_groups.extend([f'Median of {stat}', f'Mean of {stat}', f'Std of {stat}'])

            team_df = team_df[column_groups]
            player_df = player_df[column_groups]

            with open(self.output_dir / 'results2.csv', 'w', encoding='utf-8') as f:
                f.write("Team Statistics:\n")
                team_df.to_csv(f)
                f.write("\nPlayer Statistics:\n")
                player_df.to_csv(f)

            self.logger.info("Statistics saved to results2.csv")

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {str(e)}")
            raise

    def plot_distributions(self):
        try:
            sns.set(style="whitegrid")

        
            attack_stats = ['Goals', 'Sh', 'Assists']
            defense_stats = ['Tkl', 'Att', 'Blocks']
            all_stats = attack_stats + defense_stats

            pdf_path = self.plots_dir / 'all_distributions.pdf'
            with PdfPages(pdf_path) as pdf:
                for stat in all_stats:
                    if stat not in self.df.columns:
                        print(f"[SKIP] {stat} - not found in dataframe.")
                        continue
                    if not pd.api.types.is_numeric_dtype(self.df[stat]):
                        print(f"[SKIP] {stat} - not numeric.")
                        continue

                    plt.figure(figsize=(10, 5))
                    sns.histplot(data=self.df, x=stat, bins=30, kde=True, color="blue")
                    plt.title(f"Distribution of {stat} - All Players")
                    plt.xlabel(stat)
                    plt.ylabel("Frequency")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()

                    for team in self.df['Team'].dropna().unique():
                        plt.figure(figsize=(10, 5))
                        team_data = self.df[self.df['Team'] == team]
                        sns.histplot(data=team_data, x=stat, bins=20, kde=True, color="green")
                        plt.title(f"Distribution of {stat} - {team}")
                        plt.xlabel(stat)
                        plt.ylabel("Frequency")
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()

            self.logger.info(f"All distribution plots saved to {pdf_path}")

        except Exception as e:
            self.logger.error(f"Error in plot_distributions: {str(e)}")

    def identify_best_team(self):
        try:
            summary = {}
            for stat in self.numeric_columns:
                team_means = self.df.groupby('Team')[stat].mean()
                best_team = team_means.idxmax()
                best_value = team_means.max()
                summary[stat] = (best_team, best_value)

            count_top_stats = {}
            for team, _ in summary.values():
                count_top_stats[team] = count_top_stats.get(team, 0) + 1

            best_overall = max(count_top_stats.items(), key=lambda x: x[1])

            with open(self.output_dir / "team_results.txt", "w", encoding="utf-8") as f:
                f.write("Best Team by Each Statistic:\n")
                for stat, (team, value) in summary.items():
                    f.write(f"- {stat}: {team} ({value:.2f})\n")

                f.write("\nOverall Best Performing Team:\n")
                f.write(f"{best_overall[0]} leads in {best_overall[1]} stats.\n")

            self.logger.info("Best team analysis saved to team_results.txt")

        except Exception as e:
            self.logger.error(f"Error in identify_best_team: {str(e)}")

    def run_analysis(self):
        try:
            print("Loading data...")
            self.load_data()
            print("Finding top and bottom players...")
            self.find_top_bottom_players()
            print("Calculating statistics...")
            self.calculate_statistics()
            print("Creating distribution plots...")
            self.plot_distributions()
            print("Identifying best team...")
            self.identify_best_team()
            print("\nAnalysis complete! Results saved in 'analysis_output'.")
        except Exception as e:
            self.logger.error(f"Error in run_analysis: {str(e)}")
            raise

def main():
    analyzer = StatsAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
import click


@click.group()
def play():
    """Main command for playing various game strategies."""
    pass


@play.command()
def montecarlo():
    """Command for playing with Monte Carlo strategy."""
    click.echo("Monte Carlo strategy selected.")


@play.command()
def rl():
    """Command for playing with Reinforcement Learning strategy."""
    click.echo("Reinforcement Learning strategy selected.")


@play.command()
def expectimax():
    """Command for playing with Expectimax strategy."""
    click.echo("Expectimax strategy selected.")


def main():
    play()

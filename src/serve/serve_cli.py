import click
from serve import clis  


def get_command_help(command, indent=0, max_depth=None, is_last=False):
    """Recursively build help text for a command and its subcommands."""
    if max_depth is not None and indent > max_depth:
        return ""
        
    help_text = "\n"
    # Create tree branch indicators
    if indent > 0:
        prefix = "    " * (indent - 1) 
    else:
        prefix = ""
    
    cmd_help = command.help or "No description available"
    help_text += f"{prefix}{command.name:<15} {cmd_help}\n"
    
    # Add options if this is a command
    if hasattr(command, 'params') and command.params:
        for param in command.params:
            opt_names = '/'.join(opt for opt in param.opts)
            opt_help = param.help or "No description available"
            help_text += f"{prefix}    {'•':<4} {opt_names:<20} {opt_help}\n"
    
    # If this is a group, add its subcommands
    if isinstance(command, click.Group):
        subcommands = list(command.commands.items())
        for i, (subcmd_name, subcmd) in enumerate(subcommands):
            is_last_cmd = (i == len(subcommands) - 1)
            help_text += get_command_help(subcmd, indent + 1, max_depth, is_last_cmd)
            
    return help_text

def show_tree_help(ctx, param, value):
    if not value:
        return None
    
    # If value is True (flag with no number), use None for full depth
    # Otherwise, use the number provided minus 1 (since root level is 0)
    depth = None if value is True else int(value) - 1
    help_text = "serve-cmd CLI\n\n-- Available Commands --\n"
    
    # Get list of top-level commands for last-item check
    commands = list(clis)
    for i, c in enumerate(commands):
        help_text += get_command_help(c, max_depth=depth, is_last=(i == len(commands) - 1))
    
    click.echo(help_text.strip())
    ctx.exit()

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option('--help-tree', '-H', is_flag=False, is_eager=True, 
              callback=show_tree_help, flag_value=True, type=click.UNPROCESSED,
              help='Show hierarchical help. Optional: specify depth (e.g., --help-tree 2)')
def cli(help_tree=None):
    """serve-cmd CLI"""
    pass

def main():
    # Add commands and run CLI
    for c in clis:
        cli.add_command(c)
    cli()
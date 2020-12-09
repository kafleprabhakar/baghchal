import './style.scss'

function Status (props) {
    return (
        <div id="status-bar">
            <div>Turn: <span style={{color: props.turn === 'G' ? 'green': 'red'}}>{props.turn === 'G' ? 'Goat' : 'Tiger'}</span></div>
            <div>Goats Placed: {props.goats}</div>
            <div>Goats Captured: {props.goats}</div>
        </div>
    )
}

export default Status
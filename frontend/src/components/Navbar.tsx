import { useNavigate } from "react-router-dom"
import { useAuth } from "../context/AuthContext"

export default function Navbar() {
  const { username, logout } = useAuth()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate("/login")
  }

  return (
    <nav className="bg-white border-b border-gray-100 px-6 py-3 flex items-center justify-between">
      <div
        onClick={() => navigate("/library")}
        className="flex items-center gap-2 cursor-pointer group"
      >
        <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center shrink-0">
          <span className="text-sm leading-none">🎬</span>
        </div>
        <span className="font-bold text-gray-900 group-hover:text-indigo-600 transition text-sm">
          VideoMind
        </span>
      </div>

      <div className="flex items-center gap-5">
        {username && (
          <span className="text-sm text-gray-400 hidden sm:block">
            @{username}
          </span>
        )}
        <button
          onClick={() => navigate("/settings")}
          className="text-sm text-gray-500 hover:text-gray-900 transition"
        >
          Settings
        </button>
        <button
          onClick={handleLogout}
          className="text-sm text-gray-500 hover:text-gray-900 transition"
        >
          Sign out
        </button>
      </div>
    </nav>
  )
}

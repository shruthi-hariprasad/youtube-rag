import { useNavigate } from "react-router-dom"
import { useAuth } from "../context/AuthContext"

export default function Navbar() {
  const { logout } = useAuth()
  const navigate = useNavigate()

  function handleLogout() {
    logout()
    navigate("/login")
  }

  return (
    <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between">
      <div
        onClick={() => navigate("/library")}
        className="flex items-center gap-2 cursor-pointer"
      >
        <span className="text-xl">🎬</span>
        <span className="font-bold text-gray-900 text-sm">VideoMind</span>
      </div>

      <button
        onClick={handleLogout}
        className="text-sm text-gray-500 hover:text-gray-800 transition"
      >
        Sign out
      </button>
    </nav>
  )
}
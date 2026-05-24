import { useState } from "react"
import { useNavigate } from "react-router-dom"
import api from "../api/axios"
import { useAuth } from "../context/AuthContext"
import Navbar from "../components/Navbar"

export default function Settings() {
  const { username, logout } = useAuth()
  const navigate = useNavigate()

  const [currentPassword, setCurrentPassword] = useState("")
  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [pwLoading, setPwLoading] = useState(false)
  const [pwSuccess, setPwSuccess] = useState("")
  const [pwError, setPwError] = useState("")

  const [deleteLoading, setDeleteLoading] = useState(false)
  const [deleteError, setDeleteError] = useState("")

  async function handleChangePassword(e: React.FormEvent) {
    e.preventDefault()
    setPwError("")
    setPwSuccess("")
    if (newPassword !== confirmPassword) {
      setPwError("New passwords don't match")
      return
    }
    if (newPassword.length < 6) {
      setPwError("New password must be at least 6 characters")
      return
    }
    setPwLoading(true)
    try {
      await api.put("/auth/password", {
        current_password: currentPassword,
        new_password: newPassword,
      })
      setPwSuccess("Password updated successfully")
      setCurrentPassword("")
      setNewPassword("")
      setConfirmPassword("")
    } catch (err: any) {
      setPwError(err.response?.data?.detail || "Failed to update password")
    } finally {
      setPwLoading(false)
    }
  }

  async function handleDeleteAccount() {
    const confirmed = window.confirm(
      "This will permanently delete your account and all your videos. This cannot be undone. Continue?"
    )
    if (!confirmed) return
    setDeleteError("")
    setDeleteLoading(true)
    try {
      await api.delete("/auth/account")
      logout()
      navigate("/login")
    } catch (err: any) {
      setDeleteError(err.response?.data?.detail || "Failed to delete account")
      setDeleteLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />

      <div className="max-w-lg mx-auto px-6 py-8 space-y-6">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Account settings</h1>
          {username && (
            <p className="text-sm text-gray-500 mt-0.5">Logged in as <span className="font-medium text-gray-700">@{username}</span></p>
          )}
        </div>

        {/* Change password */}
        <div className="bg-white rounded-2xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 mb-4">Change password</h2>
          <form onSubmit={handleChangePassword} className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Current password</label>
              <input
                type="password"
                required
                value={currentPassword}
                onChange={e => setCurrentPassword(e.target.value)}
                className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
                placeholder="••••••••"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">New password</label>
              <input
                type="password"
                required
                value={newPassword}
                onChange={e => setNewPassword(e.target.value)}
                className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
                placeholder="••••••••"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Confirm new password</label>
              <input
                type="password"
                required
                value={confirmPassword}
                onChange={e => setConfirmPassword(e.target.value)}
                className="w-full border border-gray-200 rounded-xl px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 transition"
                placeholder="••••••••"
              />
            </div>

            {pwError && <p className="text-red-500 text-sm">{pwError}</p>}
            {pwSuccess && <p className="text-green-600 text-sm">{pwSuccess}</p>}

            <button
              type="submit"
              disabled={pwLoading}
              className="bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-5 py-2.5 rounded-xl text-sm transition disabled:opacity-60"
            >
              {pwLoading ? "Updating..." : "Update password"}
            </button>
          </form>
        </div>

        {/* Danger zone */}
        <div className="bg-white rounded-2xl shadow-sm p-6 border border-red-100">
          <h2 className="text-base font-semibold text-red-600 mb-1">Danger zone</h2>
          <p className="text-sm text-gray-500 mb-4">
            Permanently deletes your account and all videos in your library. This cannot be undone.
          </p>
          {deleteError && <p className="text-red-500 text-sm mb-3">{deleteError}</p>}
          <button
            onClick={handleDeleteAccount}
            disabled={deleteLoading}
            className="bg-red-600 hover:bg-red-700 text-white font-medium px-5 py-2.5 rounded-xl text-sm transition disabled:opacity-60"
          >
            {deleteLoading ? "Deleting..." : "Delete my account"}
          </button>
        </div>
      </div>
    </div>
  )
}
